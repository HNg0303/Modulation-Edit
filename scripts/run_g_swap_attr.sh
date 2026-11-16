#!/bin/bash
# E1: does text-derived G carry attributes? SOS + cross-attn = "a photo of a cat"; G from attribute prompts
# (black cat / young kitten / fluffy Persian / watercolor), amplified by ALPHAS; scored with CLIP.
# usage: bash scripts/run_g_swap_attr.sh       env: PYTHON SEEDS ALPHAS OUT CLIP_PATH
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}
OUT=${OUT:-output/g_swap/e1}
export CLIP_PATH=${CLIP_PATH:-weights/clip-vit-large-patch14}
[ -d "$CLIP_PATH" ] || $PYTHON -c "from huggingface_hub import snapshot_download as s; s('openai/clip-vit-large-patch14', local_dir='$CLIP_PATH', allow_patterns=['*.json','*.txt','model.safetensors'])"
$PYTHON experiments/g_swap/e1_attr.py --seeds "${SEEDS:-0,1}" --alphas "${ALPHAS:-1,5,10}" --out "$OUT"
$PYTHON experiments/g_swap/e1_eval.py "$OUT" | tee "$OUT/clip_table.md"
echo "[done] $OUT/e1_grid_seed*.jpg, clip_scores.json"
