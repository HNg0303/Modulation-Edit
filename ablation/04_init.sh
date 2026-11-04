#!/bin/bash
# Part 4: null prompt params alone vs + learned offset vs + random (untrained) offset vs offset trained from random init.
# usage: [ALPHA=<best of part 2> RANDOM_STD=0.07 TRAIN_RINIT=1] bash ablation/04_init.sh
set -euo pipefail
source "$(dirname "$0")/_env.sh"
RANDOM_STD=${RANDOM_STD:-0.07}
TRAIN_RINIT=${TRAIN_RINIT:-1}
OUT=${ROOT}/04_init
LOSS_SCALES=0-4 bash ablation/optimize_vico_offsets.sh
rinit=""
if [ "$TRAIN_RINIT" = 1 ]; then
    LOSS_SCALES=0-4 OFFSET_INIT=random INIT_STD=$RANDOM_STD bash ablation/optimize_vico_offsets.sh
    rinit=${ROOT}/offsets/ls0-4_rinit
fi
$PYTHON ablation/ablation_init.py --out_dir "$OUT" --offset_root "${ROOT}/offsets/ls0-4" --offset_root_rinit "$rinit" \
    --random_std "$RANDOM_STD" "${infer_args[@]}"
bash ablation/eval_vico.sh "$OUT"
