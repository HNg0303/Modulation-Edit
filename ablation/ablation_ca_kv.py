"""
Ablation 1: does the prompt need cross-attention? ViCo prompt "a {class} {condition}" passed to the cond rows' ca_kv
('ca-prompt') or not ('ca-null', the prompt then only enters through the adaLN blend), at each --alphas value.
alpha=0 + ca-null is the plain offset sample (no prompt at all).

    python ablation/ablation_ca_kv.py --out_dir output/ablation_vico/01_ca_kv --alphas 0,0.2
"""
import argparse

from common import Variant, add_args, run

if __name__ == '__main__':
    parser = add_args(argparse.ArgumentParser())
    parser.add_argument('--alphas', default='0,0.2')
    args = parser.parse_args()
    variants = [Variant(name=f'a{a:g}_ca-{ca}', alpha=a, blend=args.blend, ca=ca, offset_root=args.offset_root,
                        blend_scales=args.blend_scales, cfg=float(args.cfg), tau=args.tau)
                for a in map(float, args.alphas.split(',')) for ca in ('prompt', 'null')]
    run(args, variants)
