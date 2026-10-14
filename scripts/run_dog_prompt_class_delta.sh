#!/usr/bin/env bash
set -euo pipefail

# Class-relative prompt modulation: G(full prompt)-G("a dog"). Reuse the same
# ten offsets and sampling settings as the existing combined experiment.
# The default delta scales 0-4 match the stated five-scale experiment.
# Set DELTA_SCALES=all to compare with run_dog_prompt_scale_sweep.sh.
#
#   bash scripts/run_dog_prompt_class_delta.sh
#   DELTA_SCALES=all bash scripts/run_dog_prompt_class_delta.sh

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
PN="${PN:-1M}"
STEPS="${STEPS:-50}"
INFER_SEED="${INFER_SEED:-0}"
DELTA_SCALES="${DELTA_SCALES:-0-4}"
OFFSET_ROOT="${OFFSET_ROOT:-output/ada_offset/dog_prompt_modulation_${STEPS}steps_bs5_train5}"
OUTPUT_DIR="${OUTPUT_DIR:-output/dog_prompt_class_delta_${STEPS}steps_delta${DELTA_SCALES}_infer_seed${INFER_SEED}}"

command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "Python not found: $PYTHON_BIN" >&2; exit 1; }
[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || { echo 'STEPS must be a positive integer' >&2; exit 1; }
[[ "$INFER_SEED" =~ ^[0-9]+$ ]] || { echo 'INFER_SEED must be nonnegative' >&2; exit 1; }
for end_scale in 3 4; do
    for train_seed in 0 1 2 3 4; do
        path="$OFFSET_ROOT/scales_0-${end_scale}/train_seed${train_seed}/offset.pth"
        [[ -f "$path" ]] || { echo "Missing offset: $path" >&2; exit 1; }
    done
done
[[ -f weights/infinity_2b_reg.pth && -f weights/infinity_vae_d32reg.pth ]] || {
    echo 'Missing Infinity model/VAE weights under weights/' >&2; exit 1;
}
[[ -d weights/flan-t5-xl ]] || { echo 'Missing T5 weights under weights/flan-t5-xl' >&2; exit 1; }

mkdir -p "$OUTPUT_DIR"
echo "[class-delta] offset root=$OFFSET_ROOT; delta scales=$DELTA_SCALES; infer seed=$INFER_SEED"
"$PYTHON_BIN" tools/infer_dog_prompt_class_delta.py \
    --pn "$PN" --model_type infinity_2b --model_path weights/infinity_2b_reg.pth \
    --vae_type 32 --vae_path weights/infinity_vae_d32reg.pth \
    --add_lvl_embeding_only_first_block 1 --use_bit_label 1 \
    --rope2d_each_sa_layer 1 --rope2d_normalized_by_hw 2 \
    --use_scale_schedule_embedding 0 --apply_spatial_patchify 0 \
    --checkpoint_type torch --bf16 1 --text_channels 2048 \
    --text_encoder_ckpt weights/flan-t5-xl \
    --offset_root "$OFFSET_ROOT" --output_dir "$OUTPUT_DIR" \
    --infer_seed "$INFER_SEED" --expected_steps "$STEPS" \
    --class_prompt "a dog" --delta_scales "$DELTA_SCALES" --skip_existing \
    2>&1 | tee "$OUTPUT_DIR/infer.log"

echo "[done] $OUTPUT_DIR/scale_0-3 and $OUTPUT_DIR/scale_0-4"
