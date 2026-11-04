"""
Ablation 4: what the learned offset contributes on top of the null-prompt modulation (alpha / blend / ca_kv fixed):
    learned     : offset trained from zeros (the method)
    null_only   : G_null, no offset ("null prompt params" alone)
    random      : untrained N(0, random_std) offset on the same scales (norm-matched noise)
    learned_rinit: offset trained from a N(0, init_std) initialization (--offset_root_rinit, optional)

    python ablation/ablation_init.py --out_dir output/ablation_vico/04_init --offset_root_rinit output/ablation_vico/offsets/ls0-4_rinit
"""
import argparse
import os.path as osp

from common import Variant, add_args, run

if __name__ == '__main__':
    parser = add_args(argparse.ArgumentParser())
    parser.add_argument('--offset_scales', default='0-4')
    parser.add_argument('--random_std', type=float, default=0.07, help='~ RMS of trained 0-4 offsets')
    parser.add_argument('--offset_root_rinit', default='', help='offsets trained from random init; empty = skip that variant')
    parser.add_argument('--variants', default='learned,null_only,random,learned_rinit')
    args = parser.parse_args()
    common = dict(alpha=args.alpha, blend=args.blend, ca='prompt', offset_scales=args.offset_scales,
                  blend_scales=args.blend_scales, cfg=float(args.cfg), tau=args.tau)
    all_variants = {
        'learned': Variant(name='learned', offset='learned', offset_root=args.offset_root, **common),
        'null_only': Variant(name='null_only', offset='none', **common),
        'random': Variant(name=f'random_std{args.random_std:g}', offset='random', random_std=args.random_std, **common),
    }
    if args.offset_root_rinit and osp.isdir(args.offset_root_rinit):
        all_variants['learned_rinit'] = Variant(name='learned_rinit', offset='learned', offset_root=args.offset_root_rinit, **common)
    run(args, [all_variants[k] for k in args.variants.split(',') if k in all_variants])
