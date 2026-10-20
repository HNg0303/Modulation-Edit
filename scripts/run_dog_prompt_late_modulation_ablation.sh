#!/usr/bin/env bash
set -euo pipefail

# Train ten independent 0-4 offsets from the same five reference images, then
# compare two late-modulation variants at one fixed inference seed. Each prompt
# gets a 2 x 10 grid: rows = variants, columns = offset training seeds 0..9.
# Example: IMAGES_DIR=contents/dog INFER_SEED=0 bash scripts/run_dog_prompt_late_modulation_ablation.sh

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
PN="${PN:-1M}"
STEPS="${STEPS:-50}"
LR="${LR:-5e-3}"
INFER_SEED="${INFER_SEED:-0}"
EARLY_CFG="${EARLY_CFG:-1.1}"
LATE_NULL_CFG="${LATE_NULL_CFG:-1.1}"
LATE_PROMPT_CFG="${LATE_PROMPT_CFG:-2.0}"
OFFSET_ROOT="${OFFSET_ROOT:-output/ada_offset/dog_late_modulation_${STEPS}steps_bs5_train10}"
OUTPUT_DIR="${OUTPUT_DIR:-output8}"
if [[ -d contents/dog ]]; then
    DEFAULT_IMAGES_DIR=contents/dog
else
    DEFAULT_IMAGES_DIR=data/dreambooth/dataset/dog
fi
IMAGES_DIR="${IMAGES_DIR:-$DEFAULT_IMAGES_DIR}"

command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "Python not found: $PYTHON_BIN" >&2; exit 1; }
[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || { echo 'STEPS must be positive' >&2; exit 1; }
[[ "$INFER_SEED" =~ ^[0-9]+$ ]] || { echo 'INFER_SEED must be nonnegative' >&2; exit 1; }
[[ -d "$IMAGES_DIR" ]] || { echo "Missing dog images: $IMAGES_DIR" >&2; exit 1; }
image_count=$(find "$IMAGES_DIR" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' -o -iname '*.bmp' \) | wc -l | tr -d '[:space:]')
[[ "$image_count" -eq 5 ]] || { echo "Expected exactly five reference images; found $image_count" >&2; exit 1; }
[[ -f weights/infinity_2b_reg.pth && -f weights/infinity_vae_d32reg.pth ]] || {
    echo 'Missing Infinity model/VAE weights under weights/' >&2; exit 1;
}
[[ -d weights/flan-t5-xl ]] || { echo 'Missing T5 weights under weights/flan-t5-xl' >&2; exit 1; }

mkdir -p "$OUTPUT_DIR"
echo "[train] five images=$IMAGES_DIR; 10 independent offsets; scales 0-4; steps=$STEPS; lr=$LR"
for train_seed in 0 1 2 3 4 5 6 7 8 9; do
    train_dir="$OFFSET_ROOT/scales_0-4/train_seed${train_seed}"
    if [[ -f "$train_dir/offset.pth" ]]; then
        echo "[reuse] $train_dir/offset.pth (validated during inference)"
        continue
    fi
    mkdir -p "$train_dir"
    echo "[train] offset seed $train_seed/9"
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
        --loss_scales 0-4 --offset_infer_scales 0-4 \
        --log_every 1 --sample_every 0 --num_samples 1 \
        --save_offset "$train_dir/offset.pth" \
        --save_file "$train_dir/optimization_reconstruction.jpg" \
        2>&1 | tee "$train_dir/optimize.log"
done

echo "[infer] fixed inference seed=$INFER_SEED for every trained offset"
echo "[ablation] 0-4 alpha schedule, CFG=$EARLY_CFG; 5-12: G_null/CFG=$LATE_NULL_CFG vs G_prompt/CFG=$LATE_PROMPT_CFG"
"$PYTHON_BIN" tools/infer_dog_prompt_late_modulation_ablation.py \
    --pn "$PN" --model_type infinity_2b --model_path weights/infinity_2b_reg.pth \
    --vae_type 32 --vae_path weights/infinity_vae_d32reg.pth \
    --add_lvl_embeding_only_first_block 1 --use_bit_label 1 \
    --rope2d_each_sa_layer 1 --rope2d_normalized_by_hw 2 \
    --use_scale_schedule_embedding 0 --apply_spatial_patchify 0 \
    --checkpoint_type torch --bf16 1 --text_channels 2048 \
    --text_encoder_ckpt weights/flan-t5-xl \
    --offset_root "$OFFSET_ROOT" --output_dir "$OUTPUT_DIR" \
    --infer_seed "$INFER_SEED" --expected_steps "$STEPS" \
    --early_cfg "$EARLY_CFG" --late_null_cfg "$LATE_NULL_CFG" \
    --late_prompt_cfg "$LATE_PROMPT_CFG" \
    2>&1 | tee "$OUTPUT_DIR/late_modulation_ablation.log"

echo "[done] comparison.png is inside each prompt directory under $OUTPUT_DIR"
