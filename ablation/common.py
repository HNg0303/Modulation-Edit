"""
Shared code of the ViCo ablations (ablation/ablation_*.py): model loading, ViCo subjects / prompts, and a batched sampler
that controls every path through which the prompt reaches Infinity when a learned ada offset (trained on the null text) is used.

Per scale si, with G = shared_ada_lin(text_proj_for_sos(text)) (the 6 shared adaLN vectors) and D_si the offset row:
    blend 'add'   : G_cond = G_null + D_si + alpha * (G_prompt - G_null)           (alpha=1 -> G_prompt + D_si)
    blend 'convex': G_cond = (1 - alpha) * (G_null + D_si) + alpha * G_prompt     (alpha=1 -> G_prompt, no offset)
    D_si = 0 outside --offset_scales; alpha = 0 outside the blend scales (both forms then give G_null + D_si)
    ca_kv of the cond rows: T5(prompt) ('prompt') or the null text ('null'); sos token and head adaLN see the null text
    base_text 'subject': G_null / sos / head come from T5(subject name) instead (G_subj + D, for offsets trained on the subject name)
    cfg != 1: the uncond partner is G_null + D_si with null ca_kv (the plain offset row), so guidance amplifies what the prompt adds

Output layout (what CSD-VAR/evaluate_personalization.py reads):
    <out_dir>/<variant>/<subject>/prompt_XX/NN.jpg      + <out_dir>/<variant>/config.json
Prompts are ViCo's 'a {} in the jungle' templates filled with the subject name ('cat_statue' -> 'a cat statue in the
jungle', trailing digits dropped: 'brown_dog2' -> 'brown dog'); --prompt_name class gives the old class-name-only prompts,
--prompt_name none the condition only ('in the jungle', subject from the offset alone).
"""
import csv
import re
import json
import os
import os.path as osp
import sys
import time
from dataclasses import asdict, dataclass, field

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import cv2
import torch
import torch.nn.functional as F

from infinity.models.basic import CrossAttnBlock
from infinity.models.infinity import sample_with_top_k_top_p_also_inplace_modifying_logits_
from tools.infer_offset_modspace import NULL, PROMPT, encode_texts
from tools.optimize_ada_offset import get_scale_schedule, parse_scales
from tools.run_infinity import add_common_arguments, encode_prompt, load_tokenizer, load_transformer, load_visual_tokenizer

VICO_DIR = '/workspace/CSD-VAR/benchmark_data/vico_csd'
MODEL_DEFAULTS = dict(
    pn='1M', model_type='infinity_2b', model_path='weights/infinity_2b_reg.pth', vae_type=32, vae_path='weights/infinity_vae_d32reg.pth',
    add_lvl_embeding_only_first_block=1, use_bit_label=1, rope2d_each_sa_layer=1, rope2d_normalized_by_hw=2,
    use_scale_schedule_embedding=0, apply_spatial_patchify=0, checkpoint_type='torch', bf16=1, text_channels=2048,
    text_encoder_ckpt='weights/flan-t5-xl', h_div_w_template=1.0,
)


@dataclass
class Variant:
    name: str
    alpha: float = 0.0
    blend: str = 'add'                 # 'add' | 'convex' | 'pdelta' (D + alpha * (G_prompt - G_null), no G_null on offset scales) | 'addp' (G_null + D + alpha * G_prompt)
    ca: str = 'prompt'                 # cross-attention of the cond rows: 'prompt' | 'null'
    offset: str = 'learned'            # 'learned' (offset_root/<subject>/offset.pth) | 'none' (G_null only) | 'random' (untrained N(0, random_std))
    offset_root: str = ''
    offset_scales: str = '0-4'         # scales where D_si is added
    offset_weight: float = 1.0         # beta: D_si is scaled by it (G_null + beta * D_si)
    base_text: str = 'null'            # text of the base modulation G_null / sos / head adaLN: 'null' | 'subject' (T5(subject name), the
                                       # --train_prompt of a subject-text offset: G_subj + D); ca_kv of the uncond rows stays null
    blend_scales: str = 'all'          # scales where the prompt modulation is blended in
    random_std: float = 0.07           # ~ RMS of trained 0-4 offsets
    cfg: float = 1.5
    tau: float = 1.0
    extra: dict = field(default_factory=dict)


def add_args(parser):
    add_common_arguments(parser)
    parser.set_defaults(**MODEL_DEFAULTS)
    for action in parser._actions:
        if action.dest in MODEL_DEFAULTS:
            action.required = False
    parser.add_argument('--data_dir', default=VICO_DIR, help='benchmark_data/vico_csd (subjects.csv, prompts_*.txt)')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--offset_root', default='output/ablation_vico/offsets/ls0-4', help='<offset_root>/<subject>/offset.pth')
    parser.add_argument('--subjects', default='', help='comma-separated; default: all 16 ViCo subjects')
    parser.add_argument('--num_prompts', type=int, default=0, help='first N prompts per subject (0 = all 31, the protocol)')
    parser.add_argument('--num_samples', type=int, default=8, help='images per prompt (ViCo protocol: 8)')
    parser.add_argument('--batch', type=int, default=1, help='samples drawn together (x2 rows with cfg); >2 OOMs at 1M (non-flash attention), no speedup')
    parser.add_argument('--prompt_name', default='subject', choices=['subject', 'class', 'none'],
                        help="word filled into '{}': the subject name ('cat statue'), the class name only ('statue'), or none (condition only: 'in the snow')")
    parser.add_argument('--prompt_suffix', default='', help="appended to every generation prompt, e.g. ' in a realistic style' (eval keeps the ViCo prompts)")
    parser.add_argument('--null_len', type=int, default=32, help='null text length the offsets were trained with')
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--blend', default='add', choices=['add', 'convex', 'pdelta', 'addp'])
    parser.add_argument('--blend_scales', default='all')
    parser.add_argument('--top_k', type=int, default=900)
    parser.add_argument('--top_p', type=float, default=0.97)
    parser.set_defaults(cfg='1.5', tau=1.0)
    return parser


SUBJECT_NAME_FIX = {'black_cat': 'grey cat'}  # ViCo's black_cat reference images show a grey cat


def subject_name(subject):
    """'cat_statue' -> 'cat statue', 'brown_dog2' -> 'brown dog'"""
    if subject in SUBJECT_NAME_FIX:
        return SUBJECT_NAME_FIX[subject]
    return re.sub(r'\d+$', '', subject).replace('_', ' ').strip()


def fill_template(template, name):
    """name None -> condition only: 'a {} in the snow' -> 'in the snow', 'a red {}' -> 'a red'"""
    if name is None:
        return re.sub(r'^an? \{\}\s*', '', template).format('').strip()
    return template.format(name)


def read_vico(data_dir, subjects='', num_prompts=0, prompt_name='subject'):
    """-> [dict(subject, concept, class_name, prompts=[filled prompts])]"""
    with open(osp.join(data_dir, 'subjects.csv')) as f:
        rows = list(csv.DictReader(f))
    keep = set(subjects.split(',')) if subjects else None
    out = []
    for r in rows:
        if keep and r['subject'] not in keep:
            continue
        with open(osp.join(data_dir, r['prompts'])) as f:
            templates = [l.strip() for l in f if l.strip()]
        if num_prompts:
            templates = templates[:num_prompts]
        name = dict(subject=subject_name(r['subject']), none=None).get(prompt_name, r['class_name'])
        out.append(dict(subject=r['subject'], concept=r['concept'], class_name=r['class_name'],
                        prompts=[fill_template(t, name) for t in templates]))
    assert out, f'no subject selected from {data_dir}'
    return out


def load_models(args, prompts):
    """-> vae, infinity, {prompt: T5 condition}; T5 is freed before the transformer is loaded (less GPU memory)"""
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    with torch.no_grad():
        t5 = {p: encode_prompt(text_tokenizer, text_encoder, p) for p in dict.fromkeys(prompts)}
    del text_encoder
    torch.cuda.empty_cache()
    vae = load_visual_tokenizer(args)
    vae.requires_grad_(False)
    infinity = load_transformer(vae, args)
    infinity.requires_grad_(False)
    return vae, infinity, t5


def load_offset(variant, subject, subject_idx, num_scales, C, device):
    """-> (num_scales, 1, 6, C) float32, zero outside variant.offset_scales"""
    scales = parse_scales(variant.offset_scales, num_scales)
    if variant.offset == 'none':
        return torch.zeros(num_scales, 1, 6, C, device=device)
    if variant.offset == 'random':
        g = torch.Generator().manual_seed(1234 + subject_idx)  # same random offset for a subject in every run
        off = torch.randn(num_scales, 1, 6, C, generator=g) * variant.random_std
    else:
        path = osp.join(variant.offset_root, subject, 'offset.pth')
        off = torch.load(path, map_location='cpu', weights_only=False)['ada_offset'].float()
        off = off.expand(num_scales, -1, -1, -1).clone() if off.shape[0] == 1 else F.pad(off, (0, 0, 0, 0, 0, 0, 0, max(num_scales - off.shape[0], 0)))
    mask = torch.zeros(num_scales, 1, 1, 1)
    mask[scales] = 1
    return (off[:num_scales] * mask * variant.offset_weight).to(device)


@torch.no_grad()
def sample(infinity, vae, texts, offset, variant, scale_schedule, n, seed, args):
    """n samples of one prompt -> list of HW3 uint8 BGR images"""
    sos_all, gss_all, ca_all = texts
    num_scales = len(scale_schedule)
    blend_scales = set(parse_scales(variant.blend_scales, num_scales))
    off_scales = set(parse_scales(variant.offset_scales, num_scales))
    g_null, g_prompt = gss_all[NULL:NULL + 1], gss_all[PROMPT:PROMPT + 1]
    use_cfg = variant.cfg != 1 and not (variant.ca == 'null' and variant.alpha == 0)  # otherwise cond == uncond
    cond_ca = ca_all[PROMPT if variant.ca == 'prompt' else NULL]
    ca = [cond_ca] * n + ([ca_all[NULL]] * n if use_cfg else [])
    rows = len(ca)
    cu_seqlens_k = F.pad(torch.tensor([len(c) for c in ca], device=sos_all.device, dtype=torch.int32).cumsum(0, dtype=torch.int32), (1, 0))
    ca_kv = torch.cat(ca, 0), cu_seqlens_k, max(len(c) for c in ca)
    head_cond = sos_all[NULL:NULL + 1].expand(rows, -1)
    infinity.rng.manual_seed(seed)

    with torch.autocast('cuda', dtype=torch.bfloat16):
        last_stage = head_cond.unsqueeze(1) + infinity.pos_start.expand(rows, 1, -1)
        for b in infinity.unregistered_blocks:
            (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(True)
        summed_codes = 0
        for si, pn in enumerate(scale_schedule):
            base = g_null + offset[si].to(g_null)
            if variant.blend == 'pdelta' and si in off_scales:  # no G_null base on the offset scales
                base = offset[si].to(g_null).expand_as(g_null)
            a = variant.alpha if si in blend_scales else 0.
            if variant.blend == 'addp':  # G_null + D + alpha * G_prompt
                cond = base + a * g_prompt
            elif variant.blend in ('add', 'pdelta'):
                cond = base + a * (g_prompt - g_null)
            else:
                cond = (1 - a) * base + a * g_prompt
            gss_si = torch.cat([cond.expand(n, -1, -1, -1)] + ([base.expand(n, -1, -1, -1)] if use_cfg else []), 0)
            for block_idx, chunk in enumerate(infinity.block_chunks):
                if (infinity.add_lvl_embeding_only_first_block and block_idx == 0) or not infinity.add_lvl_embeding_only_first_block:
                    last_stage = infinity.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=0)
                for m in chunk.module:
                    last_stage = m(x=last_stage, cond_BD=gss_si, ca_kv=ca_kv, attn_bias_or_two_vector=None, attn_fn=None,
                                   scale_schedule=scale_schedule, rope2d_freqs_grid=infinity.rope2d_freqs_grid, scale_ind=si)
            logits = infinity.get_logits(last_stage, head_cond)
            logits = (variant.cfg * logits[:n] + (1 - variant.cfg) * logits[n:] if use_cfg else logits).mul(1 / variant.tau)
            idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(
                logits.reshape(n, -1, 2), rng=infinity.rng, top_k=args.top_k, top_p=args.top_p, num_samples=1)[:, :, 0]
            idx_Bld = idx_Bld.reshape(n, pn[1], pn[2], -1).unsqueeze(1)  # n, 1, h, w, d
            codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label')  # n, d, 1, h, w
            if si != num_scales - 1:
                summed_codes += F.interpolate(codes, size=scale_schedule[-1], mode=vae.quantizer.z_interplote_up)
                last_stage = F.interpolate(summed_codes, size=scale_schedule[si + 1], mode=vae.quantizer.z_interplote_up).squeeze(-3)
                last_stage = last_stage.reshape(*last_stage.shape[:2], -1).permute(0, 2, 1)  # n, h*w, d
                last_stage = infinity.word_embed(infinity.norm0_ve(last_stage)).repeat(rows // n, 1, 1)
            else:
                summed_codes += codes
        for b in infinity.unregistered_blocks:
            (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(False)
        img = vae.decode(summed_codes.squeeze(-3))
    img = ((img + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))
    return list(img.cpu().numpy())


def run(args, variants):
    """Generate every variant on every selected ViCo subject / prompt; resumes (complete prompt folders are skipped)."""
    args.cfg = float(args.cfg)
    subjects = read_vico(args.data_dir, args.subjects, args.num_prompts, args.prompt_name)
    for sub in subjects:
        sub['prompts'] = [p + args.prompt_suffix for p in sub['prompts']]
    vae, infinity, t5 = load_models(args, [p for sub in subjects for p in sub['prompts']] + [subject_name(sub['subject']) for sub in subjects])
    _, scale_schedule, _ = get_scale_schedule(args, 1.0)
    num_scales = len(scale_schedule)
    device = infinity.pos_start.device
    for v in variants:
        vdir = osp.join(args.out_dir, v.name)
        os.makedirs(vdir, exist_ok=True)
        with open(osp.join(vdir, 'config.json'), 'w') as f:
            json.dump(dict(variant=asdict(v), prompt_name=args.prompt_name, prompt_suffix=args.prompt_suffix, num_samples=args.num_samples, num_prompts=args.num_prompts, null_len=args.null_len,
                           pn=args.pn, top_k=args.top_k, top_p=args.top_p, seed=args.seed), f, indent=2)
        print(f'== variant {v.name}: {asdict(v)}', flush=True)
        for s_idx, sub in enumerate(subjects):
            offset = load_offset(v, sub['subject'], s_idx, num_scales, infinity.C, device)
            if v.base_text == 'subject':  # the NULL row of sos / gss (base, head, add's G_null) -> T5(subject name)
                b_sos, b_gss, _ = encode_texts(infinity, args.null_len, t5[subject_name(sub['subject'])])
            stt, made = time.time(), 0
            for p_idx, prompt in enumerate(sub['prompts']):
                pdir = osp.join(vdir, sub['subject'], f'prompt_{p_idx:02d}')
                if osp.isdir(pdir) and len([f for f in os.listdir(pdir) if f.endswith('.jpg')]) >= args.num_samples:
                    continue
                os.makedirs(pdir, exist_ok=True)
                texts = encode_texts(infinity, args.null_len, t5[prompt])
                if v.base_text == 'subject':
                    sos, gss, ca = texts
                    sos, gss = sos.clone(), gss.clone()
                    sos[NULL], gss[NULL] = b_sos[PROMPT], b_gss[PROMPT]
                    texts = sos, gss, ca
                for start in range(0, args.num_samples, args.batch):
                    n = min(args.batch, args.num_samples - start)
                    # the same seed for a (prompt, batch) in every variant -> paired comparisons
                    imgs = sample(infinity, vae, texts, offset, v, scale_schedule, n, args.seed + 1000 * p_idx + start, args)
                    for i, img in enumerate(imgs):
                        cv2.imwrite(osp.join(pdir, f'{start + i:02d}.jpg'), img)
                    made += n
            if made:
                print(f'   [{v.name}] {sub["subject"]}: {made} images, {(time.time() - stt) / made:.2f}s/img', flush=True)
