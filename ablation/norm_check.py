"""
Norm check for the 'add' blend: compare the learned offset D_si with the prompt delta d = G_prompt - G_null
(the 6 shared adaLN vectors) per subject, scale and adaLN chunk. No sampling.
    python ablation/norm_check.py --out output/ablation_vico/norm_check
-> <out>/per_subject.csv and summary.md: ||D||, ||d|| (mean over prompts), ratio ||d||/||D||, cos(D, d)
"""
import argparse
import csv
import os
import os.path as osp
import sys

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import torch
import torch.nn.functional as F

from ablation.common import Variant, add_args, load_offset, read_vico
from tools.infer_offset_modspace import NULL, PROMPT, encode_texts
from tools.optimize_ada_offset import get_scale_schedule, parse_scales
from tools.run_infinity import encode_prompt, load_tokenizer, load_transformer, load_visual_tokenizer

CHUNKS = ['gamma1', 'gamma2', 'scale1', 'scale2', 'shift1', 'shift2']  # order of the 6 shared adaLN vectors


def main():
    parser = add_args(argparse.ArgumentParser())
    parser.add_argument('--out', default='output/ablation_vico/norm_check')
    [a for a in parser._actions if a.dest == 'out_dir'][0].required = False
    args = parser.parse_args()
    subjects = read_vico(args.data_dir, args.subjects, args.num_prompts, args.prompt_name)
    vae = load_visual_tokenizer(args)
    infinity = load_transformer(vae, args)
    infinity.requires_grad_(False)
    del vae
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    _, scale_schedule, _ = get_scale_schedule(args, 1.0)
    num_scales = len(scale_schedule)
    device = infinity.pos_start.device
    v = Variant('learned')
    v.offset, v.offset_root, v.offset_scales = 'learned', args.offset_root, '0-4'
    off_scales = parse_scales(v.offset_scales, num_scales)
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for s_idx, sub in enumerate(subjects):
        off = load_offset(v, sub['subject'], s_idx, num_scales, infinity.C, device)[:, 0]  # S, 6, C
        ds = []
        for prompt in sub['prompts']:
            _, gss, _ = encode_texts(infinity, args.null_len, encode_prompt(text_tokenizer, text_encoder, prompt))
            ds.append(gss[PROMPT, 0] - gss[NULL, 0])  # 6, C
        d = torch.stack(ds)  # P, 6, C
        g_null = gss[NULL, 0]
        for si in off_scales:
            for ci, name in enumerate(CHUNKS):
                o = off[si, ci]
                rows.append(dict(subject=sub['subject'], scale=si, chunk=name, o_norm=o.norm().item(),
                                 d_norm=d[:, ci].norm(dim=-1).mean().item(), gnull_norm=g_null[ci].norm().item(),
                                 cos=F.cosine_similarity(d[:, ci], o[None], dim=-1).mean().item()))
            o, dd = off[si].flatten(), d.flatten(1)
            rows.append(dict(subject=sub['subject'], scale=si, chunk='all', o_norm=o.norm().item(), d_norm=dd.norm(dim=-1).mean().item(),
                             gnull_norm=g_null.norm().item(), cos=F.cosine_similarity(dd, o[None], dim=-1).mean().item()))
        print(f'{sub["subject"]}: done', flush=True)
    for r in rows:
        r['ratio_d_o'] = r['d_norm'] / max(r['o_norm'], 1e-12)
    with open(osp.join(args.out, 'per_subject.csv'), 'w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def mean(sel, k):
        return sum(r[k] for r in sel) / len(sel)
    keys = ['o_norm', 'd_norm', 'gnull_norm', 'ratio_d_o', 'cos']
    lines = [f'# norm check (prompt_name={args.prompt_name}, offsets {args.offset_root}), mean over {len(subjects)} subjects\n',
             '| scale | chunk | ' + ' | '.join(keys) + ' |', '|' + '---|' * (len(keys) + 2)]
    for si in off_scales:
        for name in CHUNKS + ['all']:
            sel = [r for r in rows if r['scale'] == si and r['chunk'] == name]
            lines.append(f'| {si} | {name} | ' + ' | '.join(f'{mean(sel, k):.4f}' for k in keys) + ' |')
    sel = [r for r in rows if r['chunk'] == 'all']
    lines.append(f'| 0-4 | all | ' + ' | '.join(f'{mean(sel, k):.4f}' for k in keys) + ' |')
    open(osp.join(args.out, 'summary.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
