#!/usr/bin/env bash
set -euo pipefail

# Train separate dog offsets on scales 0-3, 0-4, 0-5, and 0-6.
# For each offset: 5 prompts x 4 alpha values x 2 CFG values x 5 seeds = 200 images.
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
LR="${LR:-5e-3}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-output/ada_offset/dog_prompt_modulation_${STEPS}steps_bs5}"
DEFAULT_PROMPTS='sleeping on a bed|running through snow|wearing a red hat|seen from behind on a beach|jumping over a fallen log'
PROMPTS="${PROMPTS:-$DEFAULT_PROMPTS}"
ALPHAS="${ALPHAS:-0,0.25,0.5,1}"
CFG="${CFG:-1,1.5}"
SEED="${SEED:-0}"

command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "Python not found: $PYTHON_BIN" >&2; exit 1; }
[[ -d "$IMAGES_DIR" ]] || { echo "Dog images not found: $IMAGES_DIR" >&2; exit 1; }
[[ -f weights/infinity_2b_reg.pth && -f weights/infinity_vae_d32reg.pth ]] || {
    echo 'Infinity weights not found under weights/' >&2
    exit 1
}
[[ -d weights/flan-t5-xl ]] || { echo 'T5 weights not found: weights/flan-t5-xl' >&2; exit 1; }
read -r -a SCALE_LIST <<< "$SCALE_ENDS"
(( ${#SCALE_LIST[@]} > 0 )) || { echo 'SCALE_ENDS is empty' >&2; exit 1; }
for end_scale in "${SCALE_LIST[@]}"; do
    [[ "$end_scale" =~ ^(3|4|5|6)$ ]] || { echo "Invalid scale endpoint: $end_scale (choose 3, 4, 5, or 6)" >&2; exit 1; }
    run_dir="$EXPERIMENT_DIR/scales_0-${end_scale}"
    [[ ! -e "$run_dir/offset.pth" ]] || {
        echo "Offset already exists: $run_dir/offset.pth (choose a new EXPERIMENT_DIR)" >&2
        exit 1
    }
done

for end_scale in "${SCALE_LIST[@]}"; do
    loss_scales="0-${end_scale}"
    run_dir="$EXPERIMENT_DIR/scales_${loss_scales}"
    mkdir -p "$run_dir"
    echo "Optimizing scales $loss_scales: steps=$STEPS, batch_size=5, grad_accum=1"
    "$PYTHON_BIN" tools/optimize_ada_offset.py \
        --mode optimize --images "$IMAGES_DIR" \
        --pn "$PN" --model_type infinity_2b --model_path weights/infinity_2b_reg.pth \
        --vae_type 32 --vae_path weights/infinity_vae_d32reg.pth \
        --add_lvl_embeding_only_first_block 1 --use_bit_label 1 \
        --rope2d_each_sa_layer 1 --rope2d_normalized_by_hw 2 \
        --use_scale_schedule_embedding 0 --apply_spatial_patchify 0 \
        --checkpoint_type torch --bf16 1 --text_channels 2048 \
        --seed "$SEED" --hflip 1 --grad_checkpointing 1 --offset_mode per_scale \
        --iters "$STEPS" --batch_size 5 --grad_accum 1 --lr "$LR" \
        --loss_scales "$loss_scales" --offset_infer_scales "$loss_scales" \
        --log_every 1 --sample_every 0 --num_samples 1 \
        --save_offset "$run_dir/offset.pth" \
        --save_file "$run_dir/optimization_reconstruction.jpg" \
        2>&1 | tee "$run_dir/optimize.log"

    echo "Generating 200 images for scales $loss_scales."
    PYTHON_BIN="$PYTHON_BIN" PN="$PN" OFFSET_PATH="$run_dir/offset.pth" \
    PROMPTS="$PROMPTS" ALPHAS="$ALPHAS" CFG="$CFG" SEED="$SEED" \
    NUM_SAMPLES=5 VARIANTS=combined OUT_DIR="$run_dir/inference" \
        bash scripts/infer_offset_prompt_modulation.sh dog
done

echo "Done: $EXPERIMENT_DIR"
