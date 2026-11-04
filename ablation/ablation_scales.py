"""
Ablation 3: which scales carry the offset. One offset per --scale_sets entry, trained on those scales
(ablation/optimize_vico_offsets.sh LOSS_SCALES=<set>) and applied on the same scales; alpha / blend / ca_kv fixed.

    python ablation/ablation_scales.py --out_dir output/ablation_vico/03_scales --scale_sets 0-2,0-3,0-4,0-6 \
        --offset_root_tmpl output/ablation_vico/offsets/ls{}
"""
import argparse

from common import Variant, add_args, run

if __name__ == '__main__':
    parser = add_args(argparse.ArgumentParser())
    parser.add_argument('--scale_sets', default='0-2,0-3,0-4,0-6')
    parser.add_argument('--offset_root_tmpl', default='output/ablation_vico/offsets/ls{}', help="'{}' = scale set")
    args = parser.parse_args()
    variants = [Variant(name=f's{s}', alpha=args.alpha, blend=args.blend, ca='prompt', offset_root=args.offset_root_tmpl.format(s),
                        offset_scales=s, blend_scales=args.blend_scales, cfg=float(args.cfg), tau=args.tau)
                for s in args.scale_sets.split(',')]
    run(args, variants)
