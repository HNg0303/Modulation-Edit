"""
Ablation 2: how much prompt modulation to blend into the null modulation + offset, alpha in [0, 1] (ca_kv = prompt).
    add   : G = G_null + D + alpha * (G_prompt - G_null)
    convex: G = (1 - alpha) * (G_null + D) + alpha * G_prompt

    python ablation/ablation_blend.py --out_dir output/ablation_vico/02_blend --alphas 0,0.1,...,1 --blends add
"""
import argparse

from common import Variant, add_args, run

if __name__ == '__main__':
    parser = add_args(argparse.ArgumentParser())
    parser.add_argument('--alphas', default='0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0')
    parser.add_argument('--blends', default='add', help="comma-separated: add,convex")
    parser.add_argument('--ca', default='prompt', choices=['prompt', 'null'])
    args = parser.parse_args()
    variants = [Variant(name=f'{blend}_a{a:g}', alpha=a, blend=blend, ca=args.ca, offset_root=args.offset_root,
                        blend_scales=args.blend_scales, cfg=float(args.cfg), tau=args.tau)
                for blend in args.blends.split(',') for a in map(float, args.alphas.split(','))]
    run(args, variants)
