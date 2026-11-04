#!/bin/bash
# Part 1: prompt cross-attention on / off (ca_kv = T5(prompt) vs null) at ALPHAS; needs the ls0-4 ViCo offsets.
# usage: [ALPHAS=0,0.2 NUM_SAMPLES=8] bash ablation/01_ca_kv.sh
set -euo pipefail
source "$(dirname "$0")/_env.sh"
ALPHAS=${ALPHAS:-0,0.2}
OUT=${ROOT}/01_ca_kv
LOSS_SCALES=0-4 bash ablation/optimize_vico_offsets.sh
$PYTHON ablation/ablation_ca_kv.py --out_dir "$OUT" --offset_root "${ROOT}/offsets/ls0-4" --alphas "$ALPHAS" "${infer_args[@]}"
bash ablation/eval_vico.sh "$OUT"
