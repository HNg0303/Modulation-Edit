#!/usr/bin/env bash
set -euo pipefail

# Prompt + learned subject offset; PROMPT_CA_KV=0 keeps cross-attention null.
#   trained scales: G_null + D_s + alpha * (G_prompt - G_null)
#   other scales:   G_null       + alpha * (G_prompt - G_null)
# The CFG partner has the same subject offset but null CA and null AdaLN.
# Usage: bash scripts/infer_offset_prompt_modulation.sh dog
# Example: PROMPTS='a dog sleeping|a dog wearing a red hat' ALPHAS=0,0.25,0.5 CFG=3 \
#          bash scripts/infer_offset_prompt_modulation.sh dog
# For a single checkpoint: OFFSET_PATH=output/ada_offset_hybrid/dog_ls0-3/offset.pth \
#          bash scripts/infer_offset_prompt_modulation.sh dog

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
PN="${PN:-1M}"
RUN_DIR="${RUN_DIR:-output/ada_offset/dreambooth_ls0-6_bs1}"
OFFSET_PATH="${OFFSET_PATH:-}"
ITER="${ITER:-100}"
DEFAULT_PROMPTS='a {} sleeping|a {} wearing a red hat'
PROMPTS="${PROMPTS:-$DEFAULT_PROMPTS}"
ALPHAS="${ALPHAS:-0,0.25,0.5,1}"
CFG="${CFG:-3}"
TAU="${TAU:-1}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
VARIANTS="${VARIANTS:-offset,combined}"
PROMPT_CA_KV="${PROMPT_CA_KV:-1}"
SEED="${SEED:-0}"
[[ "$PROMPT_CA_KV" == 0 || "$PROMPT_CA_KV" == 1 ]] || { echo 'PROMPT_CA_KV must be 0 or 1' >&2; exit 1; }
if [[ -n "$OFFSET_PATH" ]]; then
    DEFAULT_OUT_DIR="$(dirname "$OFFSET_PATH")/prompt_modulation_ca${PROMPT_CA_KV}"
    SOURCE_ARGS=(--offset_path "$OFFSET_PATH" --iter "$ITER")
else
    DEFAULT_OUT_DIR="${RUN_DIR}/prompt_modulation_ca${PROMPT_CA_KV}_it$(printf '%05d' "$ITER")"
    SOURCE_ARGS=(--run_dir "$RUN_DIR" --iter "$ITER")
fi
OUT_DIR="${OUT_DIR:-$DEFAULT_OUT_DIR}"
CLASSES="${CLASSES:-data/dreambooth/dataset/prompts_and_classes.txt}"

command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "Python not found: $PYTHON_BIN" >&2; exit 1; }
if [[ -n "$OFFSET_PATH" ]]; then
    [[ -f "$OFFSET_PATH" ]] || { echo "Offset not found: $OFFSET_PATH" >&2; exit 1; }
else
    [[ -d "$RUN_DIR" ]] || { echo "Run directory not found: $RUN_DIR" >&2; exit 1; }
fi
[[ -f weights/infinity_2b_reg.pth && -f weights/infinity_vae_d32reg.pth ]] || {
    echo 'Model weights not found under weights/' >&2
    exit 1
}

subjects=$(IFS=,; echo "$*")
mkdir -p "$OUT_DIR"

"$PYTHON_BIN" tools/infer_offset_modspace.py \
    --pn "$PN" --model_type infinity_2b --model_path weights/infinity_2b_reg.pth \
    --vae_type 32 --vae_path weights/infinity_vae_d32reg.pth \
    --add_lvl_embeding_only_first_block 1 --use_bit_label 1 \
    --rope2d_each_sa_layer 1 --rope2d_normalized_by_hw 2 \
    --use_scale_schedule_embedding 0 --apply_spatial_patchify 0 \
    --checkpoint_type torch --bf16 1 --text_channels 2048 \
    --text_encoder_ckpt weights/flan-t5-xl \
    "${SOURCE_ARGS[@]}" --subjects "$subjects" --classes "$CLASSES" \
    --prompts "$PROMPTS" --delta_weights "$ALPHAS" --delta_scales all --prompt_ca_kv "$PROMPT_CA_KV" \
    --offset_scales all --variants "$VARIANTS" \
    --cfg "$CFG" --tau "$TAU" --seed "$SEED" --num_samples "$NUM_SAMPLES" \
    --out_dir "$OUT_DIR" \
    2>&1 | tee "$OUT_DIR/infer.log"

echo "Saved results to $OUT_DIR"
