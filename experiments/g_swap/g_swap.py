"""G-swap experiment on pretrained Infinity-2B (no optimization).

Conditioning routes in Infinity:
  pooled text  -> cond_BD -> (a) SOS token  (b) G = shared_ada_lin(cond_BD)  (AdaLN gamma/scale/shift, all blocks)
  token text   -> text_proj_for_ca -> cross-attention K/V
Here SOS + cross-attention (and the head's AdaLN, which reads cond_BD, not G) always come from the
neutral prompt. Only G is replaced, per scale, via a forward pre-hook on every block.
CFG: only the conditional half of the batch is modified; the unconditional half keeps its own (null) G.
"""
import os, sys, argparse, json
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(HERE, 'shim'))   # pure-torch flash_attn stand-in
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch, numpy as np, cv2
from types import SimpleNamespace
from tools.run_infinity import load_tokenizer, load_visual_tokenizer, load_transformer, encode_prompt
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w

p = argparse.ArgumentParser()
p.add_argument('--neutral', default='a photo')
p.add_argument('--c1', default='cat')
p.add_argument('--c2', default='dog')
p.add_argument('--g_template', default='a photo of a {}', help='prompt used to compute G_c')
p.add_argument('--seeds', default='0,1,2,3')
p.add_argument('--early', default='4', help='|S| for the interpolation sweep (first k scales)')
p.add_argument('--ts', default='0,0.25,0.5,0.75,1')
p.add_argument('--alphas', default='5,20', help='amplification of (G_c - G_n), extra control')
p.add_argument('--cfg', type=float, default=4.0)
p.add_argument('--tau', type=float, default=0.5)
p.add_argument('--pn', default='1M')
p.add_argument('--out', default=os.path.join(REPO, 'output/g_swap/e0'))
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

margs = SimpleNamespace(
    model_path=f'{REPO}/weights/infinity_2b_reg.pth', vae_path=f'{REPO}/weights/infinity_vae_d32reg.pth',
    text_encoder_ckpt=f'{REPO}/weights/flan-t5-xl', vae_type=32, model_type='infinity_2b', pn=args.pn,
    add_lvl_embeding_only_first_block=1, use_bit_label=1, rope2d_each_sa_layer=1, rope2d_normalized_by_hw=2,
    use_scale_schedule_embedding=0, text_channels=2048, apply_spatial_patchify=0, use_flex_attn=0, bf16=1,
    checkpoint_type='torch', cache_dir='/nonexistent', enable_model_cache=0)
tok, enc = load_tokenizer(margs.text_encoder_ckpt)
vae = load_visual_tokenizer(margs)
model = load_transformer(vae, margs)
scale_schedule = [(1, h, w) for (_, h, w) in dynamic_resolution_h_w[1.0][args.pn]['scales']]
NS = len(scale_schedule)
print('scale schedule:', scale_schedule)


@torch.no_grad()
def G_of(prompt):
    kv, lens, cu, Lmax = encode_prompt(tok, enc, prompt)
    kv = model.text_norm(kv)
    cond = model.text_proj_for_sos((kv, cu, Lmax))
    with torch.autocast('cuda', enabled=False):
        return model.shared_ada_lin(cond.float()).float()     # [1,1,6,C]


# ---- per-scale G override via block pre-hooks ----
STATE = {'G': None}   # G: callable si -> tensor [1,1,6,C] or None (= keep)
def pre_hook(mod, a, kw):
    f = STATE['G']
    if f is None: return None
    g = f(kw['scale_ind'])
    if g is None: return None
    cond = kw['cond_BD'].clone()
    cond[:g.shape[0]] = g.to(cond.dtype)       # cond half only; uncond half untouched
    kw['cond_BD'] = cond
    return a, kw
for b in model.unregistered_blocks:
    b.register_forward_pre_hook(pre_hook, with_kwargs=True)


@torch.no_grad()
def generate(prompt, seed, Gfn=None):
    STATE['G'] = Gfn
    cond = encode_prompt(tok, enc, prompt)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        _, _, img_list = model.autoregressive_infer_cfg(
            vae=vae, scale_schedule=scale_schedule, label_B_or_BLT=cond, g_seed=seed, B=1,
            negative_label_B_or_BLT=None, force_gt_Bhw=None, cfg_sc=3,
            cfg_list=[args.cfg] * NS, tau_list=[args.tau] * NS, top_k=900, top_p=0.97,
            returns_vemb=1, ratio_Bl1=None, gumbel=0, norm_cfg=False, cfg_exp_k=0.0,
            cfg_insertion_layer=[0], vae_type=32, softmax_merge_topk=-1, ret_img=True,
            trunk_scale=1000, gt_leak=0, gt_ls_Bl=None, inference_mode=True, sampling_per_bits=1)
    STATE['G'] = None
    return img_list[0].cpu().numpy()    # HWC uint8, BGR (as in run_infinity)


def save(img, name):
    cv2.imwrite(os.path.join(args.out, name), img)


G_n = G_of(args.neutral)
G1 = G_of(args.g_template.format(args.c1))
G2 = G_of(args.g_template.format(args.c2))
cos = lambda a, b: torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), 0).item()
stats = {'cos(G_c1,G_c2)': cos(G1, G2), 'cos(G_n,G_c1)': cos(G_n, G1), 'cos(G_n,G_c2)': cos(G_n, G2),
         '|G_c1-G_c2|/|G_n|': ((G1 - G2).norm() / G_n.norm()).item()}
print(stats)

seeds = [int(s) for s in args.seeds.split(',')]
ts = [float(t) for t in args.ts.split(',')]
ks = [int(k) for k in args.early.split(',')]
meta = dict(vars(args), scale_schedule=scale_schedule, G_stats=stats, files={})

for seed in seeds:
    # references: full prompt through all routes / neutral only
    for tag, pr in [('ref_neutral', args.neutral), ('ref_c1', args.g_template.format(args.c1)),
                    ('ref_c2', args.g_template.format(args.c2))]:
        save(generate(pr, seed), f's{seed}_{tag}.jpg')
    # G swapped at ALL scales, prompt fixed to neutral
    save(generate(args.neutral, seed, lambda si: G1), f's{seed}_Gall_c1.jpg')
    save(generate(args.neutral, seed, lambda si: G2), f's{seed}_Gall_c2.jpg')
    # complement control: class prompt drives SOS + cross-attn, G pinned to neutral
    save(generate(args.g_template.format(args.c1), seed, lambda si: G_n), f's{seed}_promptc1_Gn.jpg')
    save(generate(args.g_template.format(args.c2), seed, lambda si: G_n), f's{seed}_promptc2_Gn.jpg')
    # amplified class direction in G-space (all scales), prompt neutral
    for a in [float(x) for x in args.alphas.split(',') if x]:
        for tag, Gc in [('c1', G1), ('c2', G2)]:
            Ga = G_n + a * (Gc - G_n)
            save(generate(args.neutral, seed, lambda si, Ga=Ga: Ga), f's{seed}_Gamp{a:g}_{tag}.jpg')
    # interpolation G(t) on early scales S = first k scales; later scales keep neutral G
    for k in ks:
        for t in ts:
            Gt = (1 - t) * G1 + t * G2
            save(generate(args.neutral, seed, lambda si, Gt=Gt, k=k: Gt if si < k else None),
                 f's{seed}_early{k}_t{t:.2f}.jpg')
    print('done seed', seed, flush=True)

json.dump(meta, open(os.path.join(args.out, 'meta.json'), 'w'), indent=1, default=str)
