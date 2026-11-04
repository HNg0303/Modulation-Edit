#!/bin/bash
# Part 2: blend weight alpha of the prompt adaLN params, 0 -> 1 (ca_kv = prompt); BLENDS=add,convex for both forms.
# usage: [ALPHAS=0,0.1,...,1 BLENDS=add] bash ablation/02_blend.sh
set -euo pipefail
source "$(dirname "$0")/_env.sh"
ALPHAS=${ALPHAS:-0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}
BLENDS=${BLENDS:-add}
OUT=${ROOT}/02_blend
LOSS_SCALES=0-4 bash ablation/optimize_vico_offsets.sh
$PYTHON ablation/ablation_blend.py --out_dir "$OUT" --offset_root "${ROOT}/offsets/ls0-4" --alphas "$ALPHAS" --blends "$BLENDS" "${infer_args[@]}"
bash ablation/eval_vico.sh "$OUT"
