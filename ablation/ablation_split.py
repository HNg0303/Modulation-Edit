"""
Ablation 8: offset scales keep G_null + beta * D (no prompt adaLN), the other scales use G_prompt as plain Infinity does
(blend 'add' with alpha = 1 on --prompt_scales, where D = 0 -> G_prompt); ca_kv = prompt.

    python ablation/ablation_split.py --out_dir output/ablation_vico/08_split --betas 0.85 --prompt_scales 5-99
"""
import argparse

from common import Variant, add_args, run

if __name__ == '__main__':
    parser = add_args(argparse.ArgumentParser())
    parser.add_argument('--betas', default='0.85')
    parser.add_argument('--offset_scales', default='0-4')
    parser.add_argument('--prompt_scales', default='5-99')
    args = parser.parse_args()
    variants = [Variant(name=f's{args.offset_scales}_b{b:g}_p{args.prompt_scales}', alpha=1., blend='add', ca='prompt',
                        offset_root=args.offset_root, offset_scales=args.offset_scales, offset_weight=b,
                        blend_scales=args.prompt_scales, cfg=float(args.cfg), tau=args.tau)
                for b in map(float, args.betas.split(','))]
    run(args, variants)
