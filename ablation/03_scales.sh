#!/bin/bash
# Part 3 (slowest, run last): offset scale range; one offset set is optimized per range, then sampled at ALPHA.
# usage: [SCALE_SETS="0-2 0-3 0-4 0-6" ALPHA=<best of part 2>] bash ablation/03_scales.sh
set -euo pipefail
source "$(dirname "$0")/_env.sh"
SCALE_SETS=${SCALE_SETS:-0-2 0-3 0-4 0-6}
OUT=${ROOT}/03_scales
for s in $SCALE_SETS; do LOSS_SCALES=$s bash ablation/optimize_vico_offsets.sh; done
$PYTHON ablation/ablation_scales.py --out_dir "$OUT" --scale_sets "$(echo $SCALE_SETS | tr ' ' ',')" \
    --offset_root_tmpl "${ROOT}/offsets/ls{}" "${infer_args[@]}"
bash ablation/eval_vico.sh "$OUT"
