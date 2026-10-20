#!/usr/bin/env bash
set -euo pipefail

# Train five dog offsets for each scale range (one independent optimization per seed).
# Test every offset with 5 prompts x 4 alpha values x 2 CFG values x ONE fixed inference seed.
# Example: bash scripts/run_dog_prompt_modulation_experiment.sh

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
PN="${PN:-1M}"
if [[ -d contents/dog ]]; then
    DEFAULT_IMAGES_DIR=contents/dog
else
    DEFAULT_IMAGES_DIR=data/dreambooth/dataset/dog
fi
IMAGES_DIR="${IMAGES_DIR:-$DEFAULT_IMAGES_DIR}"
STEPS="${STEPS:-50}"
SCALE_ENDS="${SCALE_ENDS:-3 4 5 6}"
TRAIN_SEEDS="${TRAIN_SEEDS:-0 1 2 3 4}"
INFER_SEED="${INFER_SEED:-0}"
PROMPT_CA_KV="${PROMPT_CA_KV:-1}"
PROMPT_MOD_DELTA="${PROMPT_MOD_DELTA:-1}"
LR="${LR:-5e-3}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-output/ada_offset/dog_prompt_modulation_${STEPS}steps_bs5_train5}"
DEFAULT_PROMPTS='a dog sleeping on a bed|a dog running through snow|a dog wearing a red hat|a dog seen from behind on a beach|a dog jumping over a fallen log'
PROMPTS="${PROMPTS:-$DEFAULT_PROMPTS}"
ALPHAS="${ALPHAS:-0.1,0.2,0.3,0.4,0.5}"
CFG="${CFG:-1,1.5}"

command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "Python not found: $PYTHON_BIN" >&2; exit 1; }
[[ -d "$IMAGES_DIR" ]] || { echo "Dog images not found: $IMAGES_DIR" >&2; exit 1; }
image_count=$(find "$IMAGES_DIR" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' -o -iname '*.bmp' \) | wc -l | tr -d '[:space:]')
[[ "$image_count" -eq 5 ]] || { echo "Expected exactly 5 original images in $IMAGES_DIR; found $image_count" >&2; exit 1; }
[[ -f weights/infinity_2b_reg.pth && -f weights/infinity_vae_d32reg.pth ]] || {
    echo 'Infinity weights not found under weights/' >&2
    exit 1
}
[[ -d weights/flan-t5-xl ]] || { echo 'T5 weights not found: weights/flan-t5-xl' >&2; exit 1; }
read -r -a SCALE_LIST <<< "$SCALE_ENDS"
read -r -a TRAIN_SEED_LIST <<< "$TRAIN_SEEDS"
(( ${#SCALE_LIST[@]} > 0 )) || { echo 'SCALE_ENDS is empty' >&2; exit 1; }
(( ${#TRAIN_SEED_LIST[@]} > 0 )) || { echo 'TRAIN_SEEDS is empty' >&2; exit 1; }
for end_scale in "${SCALE_LIST[@]}"; do
    [[ "$end_scale" =~ ^(3|4|5|6)$ ]] || { echo "Invalid scale endpoint: $end_scale (choose 3, 4, 5, or 6)" >&2; exit 1; }
    for train_seed in "${TRAIN_SEED_LIST[@]}"; do
        [[ "$train_seed" =~ ^[0-9]+$ ]] || { echo "Invalid training seed: $train_seed" >&2; exit 1; }
        run_dir="$EXPERIMENT_DIR/scales_0-${end_scale}/train_seed${train_seed}"
        [[ ! -e "$run_dir/offset.pth" ]] || {
            echo "Offset already exists: $run_dir/offset.pth (choose a new EXPERIMENT_DIR)" >&2
            exit 1
        }
    done
done
[[ "$INFER_SEED" =~ ^[0-9]+$ ]] || { echo "Invalid inference seed: $INFER_SEED" >&2; exit 1; }
[[ "$PROMPT_CA_KV" == 0 || "$PROMPT_CA_KV" == 1 ]] || { echo 'PROMPT_CA_KV must be 0 or 1' >&2; exit 1; }
[[ "$PROMPT_MOD_DELTA" == 0 || "$PROMPT_MOD_DELTA" == 1 ]] || { echo 'PROMPT_MOD_DELTA must be 0 or 1' >&2; exit 1; }

for end_scale in "${SCALE_LIST[@]}"; do
    loss_scales="0-${end_scale}"
    for train_seed in "${TRAIN_SEED_LIST[@]}"; do
        run_dir="$EXPERIMENT_DIR/scales_${loss_scales}/train_seed${train_seed}"
        mkdir -p "$run_dir"
        echo "Optimizing scales $loss_scales, training seed $train_seed: steps=$STEPS, one batch of all 5 originals per step, no augmentation"
        "$PYTHON_BIN" tools/optimize_ada_offset.py \
            --mode optimize --images "$IMAGES_DIR" \
            --pn "$PN" --model_type infinity_2b --model_path weights/infinity_2b_reg.pth \
            --vae_type 32 --vae_path weights/infinity_vae_d32reg.pth \
            --add_lvl_embeding_only_first_block 1 --use_bit_label 1 \
            --rope2d_each_sa_layer 1 --rope2d_normalized_by_hw 2 \
            --use_scale_schedule_embedding 0 --apply_spatial_patchify 0 \
            --checkpoint_type torch --bf16 1 --text_channels 2048 \
            --seed "$train_seed" --hflip 0 --noise_apply_layers 0 --grad_checkpointing 1 --offset_mode per_scale \
            --iters "$STEPS" --batch_size 5 --grad_accum 1 --lr "$LR" \
            --loss_scales "$loss_scales" --offset_infer_scales "$loss_scales" \
            --log_every 1 --sample_every 0 --num_samples 1 \
            --save_offset "$run_dir/offset.pth" \
            --save_file "$run_dir/optimization_reconstruction.jpg" \
            2>&1 | tee "$run_dir/optimize.log"

        echo "Testing offset from training seed $train_seed with one fixed inference seed $INFER_SEED."
        PYTHON_BIN="$PYTHON_BIN" PN="$PN" ITER="$STEPS" OFFSET_PATH="$run_dir/offset.pth" \
        PROMPTS="$PROMPTS" ALPHAS="$ALPHAS" CFG="$CFG" SEED="$INFER_SEED" \
        PROMPT_CA_KV="$PROMPT_CA_KV" PROMPT_MOD_DELTA="$PROMPT_MOD_DELTA" \
        NUM_SAMPLES=1 VARIANTS=combined OUT_DIR="$run_dir/inference_ca${PROMPT_CA_KV}_mod${PROMPT_MOD_DELTA}" \
            bash scripts/infer_offset_prompt_modulation.sh dog
    done
done

echo "Done: $EXPERIMENT_DIR"
