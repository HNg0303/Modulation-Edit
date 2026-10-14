#!/usr/bin/env bash
set -euo pipefail

# Two scale ranges x five independently trained offsets x five prompts.
# Each prompt comparison is four rows (alpha 0.1/0.2 x CFG 1/1.5)
# by five columns (offset training seeds 0..4). Inference seed stays fixed.
#
#   bash scripts/run_dog_prompt_scale_sweep.sh
#   INFER_SEED=2 bash scripts/run_dog_prompt_scale_sweep.sh
#   OFFSET_ROOT=/path/to/previous/run bash scripts/run_dog_prompt_scale_sweep.sh

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
PN="${PN:-1M}"
STEPS="${STEPS:-50}"
LR="${LR:-5e-3}"
INFER_SEED="${INFER_SEED:-0}"
OFFSET_ROOT="${OFFSET_ROOT:-output/ada_offset/dog_prompt_modulation_${STEPS}steps_bs5_train5}"
OUTPUT_DIR="${OUTPUT_DIR:-output/dog_prompt_scale_sweep_${STEPS}steps_infer_seed${INFER_SEED}}"

if [[ -d contents/dog ]]; then
    DEFAULT_IMAGES_DIR=contents/dog
else
    DEFAULT_IMAGES_DIR=data/dreambooth/dataset/dog
fi
IMAGES_DIR="${IMAGES_DIR:-$DEFAULT_IMAGES_DIR}"

command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "Python not found: $PYTHON_BIN" >&2; exit 1; }
[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || { echo 'STEPS must be a positive integer' >&2; exit 1; }
[[ "$INFER_SEED" =~ ^[0-9]+$ ]] || { echo 'INFER_SEED must be nonnegative' >&2; exit 1; }
[[ -d "$IMAGES_DIR" ]] || { echo "Missing dog images: $IMAGES_DIR" >&2; exit 1; }
image_count=$(find "$IMAGES_DIR" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' -o -iname '*.bmp' \) | wc -l | tr -d '[:space:]')
[[ "$image_count" -eq 5 ]] || { echo "Expected exactly five dog images; found $image_count" >&2; exit 1; }
[[ -f weights/infinity_2b_reg.pth && -f weights/infinity_vae_d32reg.pth ]] || {
    echo 'Infinity model/VAE weights not found under weights/' >&2; exit 1;
}
[[ -d weights/flan-t5-xl ]] || { echo 'T5 weights not found: weights/flan-t5-xl' >&2; exit 1; }

for end_scale in 3 4; do
    for train_seed in 0 1 2 3 4; do
        run_dir="$OFFSET_ROOT/scales_0-${end_scale}/train_seed${train_seed}"
        if [[ -f "$run_dir/offset.pth" ]]; then
            echo "[reuse] $run_dir/offset.pth"
            continue
        fi
        mkdir -p "$run_dir"
        echo "[train] scales 0-$end_scale; seed $train_seed; $STEPS steps; batch size 5"
        "$PYTHON_BIN" tools/optimize_ada_offset.py \
            --mode optimize --images "$IMAGES_DIR" \
            --pn "$PN" --model_type infinity_2b --model_path weights/infinity_2b_reg.pth \
            --vae_type 32 --vae_path weights/infinity_vae_d32reg.pth \
            --add_lvl_embeding_only_first_block 1 --use_bit_label 1 \
            --rope2d_each_sa_layer 1 --rope2d_normalized_by_hw 2 \
            --use_scale_schedule_embedding 0 --apply_spatial_patchify 0 \
            --checkpoint_type torch --bf16 1 --text_channels 2048 \
            --seed "$train_seed" --hflip 0 --noise_apply_layers 0 \
            --grad_checkpointing 1 --offset_mode per_scale \
            --iters "$STEPS" --batch_size 5 --grad_accum 1 --lr "$LR" \
            --loss_scales "0-${end_scale}" --offset_infer_scales "0-${end_scale}" \
            --log_every 1 --sample_every 0 --num_samples 1 \
            --save_offset "$run_dir/offset.pth" \
            --save_file "$run_dir/optimization_reconstruction.jpg" \
            2>&1 | tee "$run_dir/optimize.log"
    done
done

mkdir -p "$OFFSET_ROOT"
echo "[infer] fixed inference seed $INFER_SEED; alpha 0.1/0.2 on all 13 scales; CFG 1/1.5"
"$PYTHON_BIN" tools/infer_dog_prompt_scale_sweep.py \
    --pn "$PN" --model_type infinity_2b --model_path weights/infinity_2b_reg.pth \
    --vae_type 32 --vae_path weights/infinity_vae_d32reg.pth \
    --add_lvl_embeding_only_first_block 1 --use_bit_label 1 \
    --rope2d_each_sa_layer 1 --rope2d_normalized_by_hw 2 \
    --use_scale_schedule_embedding 0 --apply_spatial_patchify 0 \
    --checkpoint_type torch --bf16 1 --text_channels 2048 \
    --text_encoder_ckpt weights/flan-t5-xl \
    --offset_root "$OFFSET_ROOT" --output_dir "$OUTPUT_DIR" \
    --scale_ends 3,4 --train_seeds 0,1,2,3,4 \
    --infer_seed "$INFER_SEED" --expected_steps "$STEPS" --skip_existing \
    2>&1 | tee "$OFFSET_ROOT/dog_prompt_scale_sweep_infer.log"

echo "[done] $OUTPUT_DIR/scale_0-3 and $OUTPUT_DIR/scale_0-4"
