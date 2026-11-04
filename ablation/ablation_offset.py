"""
Ablation 5: offset placement x strength with G_null kept intact (alpha = 0: G = G_null + beta * D_si on --offset_scales,
G_null elsewhere; ca_kv = prompt). Placement masks the ls0-4 offset to a scale subset (no retraining).

    python ablation/ablation_offset.py --out_dir output/ablation_vico/05_offset --scale_sets 0-4,0-2 --betas 0.5,1
"""
import argparse

from common import Variant, add_args, run

if __name__ == '__main__':
    parser = add_args(argparse.ArgumentParser())
    parser.add_argument('--scale_sets', default='0-1;0-2;0-3;0-4', help="';'-separated, e.g. '2-4;0,2-4'")
    parser.add_argument('--betas', default='0.5,0.7,0.85,1')
    parser.add_argument('--tag', default='', help='variant name suffix, e.g. the offset set')
    args = parser.parse_args()
    variants = [Variant(name=f's{s}_b{b:g}{args.tag}', alpha=0., ca='prompt', offset_root=args.offset_root, offset_scales=s,
                        offset_weight=b, cfg=float(args.cfg), tau=args.tau)
                for s in args.scale_sets.split(';') for b in map(float, args.betas.split(','))]
    run(args, variants)
