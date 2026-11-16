#!/bin/bash
# E3: does a learned early-scale (A+-like) adaLN offset need a base concept? Train a per-scale offset on scales
# LOSS_SCALES of one DreamBooth subject, then sample it (x0 and x1) on top of different base prompts.
# usage: bash scripts/run_offset_concept.sh [subject]   env: PYTHON PN ITERS LOSS_SCALES TRAIN_PROMPT PROMPTS OFFSET_SCALES NUM_SAMPLES OUT
# needs data/dreambooth/dataset/<subject>/ (e.g. curl the 5 images from github.com/google/dreambooth/tree/main/dataset/dog)
set -euo pipefail
cd "$(dirname "$0")/.."
subject=${1:-dog}
PYTHON=${PYTHON:-python}
PN=${PN:-0.25M}
ITERS=${ITERS:-60}
LOSS_SCALES=${LOSS_SCALES:-0-5}
TRAIN_PROMPT=${TRAIN_PROMPT:-a dog}
PROMPTS=${PROMPTS:-a dog|a photo|a photo of a cat|a dog on the beach|a dog in a bucket}
OFFSET_SCALES=${OFFSET_SCALES:-0,1}
OUT=${OUT:-output/g_swap/e3}
export PYTHONPATH=experiments/g_swap/shim${PYTHONPATH:+:$PYTHONPATH}   # pure-torch flash_attn stand-in (no-op if real flash_attn is first)
mkdir -p "$OUT"
common=(
    --pn "$PN" --model_type infinity_2b --model_path weights/infinity_2b_reg.pth
    --vae_type 32 --vae_path weights/infinity_vae_d32reg.pth
    --add_lvl_embeding_only_first_block 1 --use_bit_label 1 --rope2d_each_sa_layer 1 --rope2d_normalized_by_hw 2
    --use_scale_schedule_embedding 0 --apply_spatial_patchify 0 --checkpoint_type torch --bf16 1
    --text_encoder_ckpt weights/flan-t5-xl --text_channels 2048 --seed 0 --offset_infer_scales "$LOSS_SCALES"
)
$PYTHON tools/optimize_ada_offset.py --mode optimize "${common[@]}" \
    --images "data/dreambooth/dataset/$subject" --hflip 1 --grad_checkpointing 1 --offset_mode per_scale --train_prompt "$TRAIN_PROMPT" \
    --iters "$ITERS" --batch_size 2 --grad_accum 1 --lr 5e-3 --loss_scales "$LOSS_SCALES" --log_every 10 --num_samples 2 \
    --save_offset "$OUT/offset.pth" --save_file "$OUT/train_samples.jpg" 2>&1 | tee "$OUT/optimize.log"
$PYTHON tools/optimize_ada_offset.py --mode infer "${common[@]}" \
    --offset_path "$OUT/offset.pth" --offset_scale "$OFFSET_SCALES" --num_samples "${NUM_SAMPLES:-3}" \
    --prompt "$PROMPTS" --save_file "$OUT/e3_infer.jpg" 2>&1 | tee "$OUT/infer.log" | grep -E '^\[(infer|saved)\]|Error' || true
echo "[done] $OUT/e3_infer.jpg  (rows: offset_scale x prompt, last row = training images)"
