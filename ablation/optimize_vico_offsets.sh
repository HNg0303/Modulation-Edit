#!/bin/bash
# Optimize a per-scale ada offset (null text + offset, teacher forcing) for every ViCo subject.
# usage: LOSS_SCALES=0-4 [OFFSET_INIT=random INIT_STD=0.07 SUBJECTS=a,b] bash ablation/optimize_vico_offsets.sh [subject ...]
# -> ${ROOT}/offsets/ls${LOSS_SCALES}[_rinit]/<subject>/{offset.pth, optimize.log}; done subjects are skipped
set -uo pipefail
source "$(dirname "$0")/_env.sh"
LOSS_SCALES=${LOSS_SCALES:-0-4}
OFFSET_INIT=${OFFSET_INIT:-zeros}
INIT_STD=${INIT_STD:-0.07}
ITERS=${ITERS:-100}
BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-2}
LR=${LR:-5e-3}
suffix=$([ "$OFFSET_INIT" = random ] && echo _rinit || true)
OUT=${OUT:-${ROOT}/offsets/ls${LOSS_SCALES}${suffix}}

if [ $# -gt 0 ]; then subjects=("$@"); elif [ -n "$SUBJECTS" ]; then IFS=, read -ra subjects <<< "$SUBJECTS"; else mapfile -t subjects < <(tail -n +2 "$VICO_DIR/subjects.csv" | cut -d, -f1); fi
failed=()
for concept in "${subjects[@]}"; do
    subject=${concept%%+*}
    dst=${OUT}/${subject}
    [ -f "$dst/offset.pth" ] && { echo "== skip ${subject}: ${dst}/offset.pth exists"; continue; }
    mkdir -p "$dst"
    echo "== [${subject}] offset on scales ${LOSS_SCALES} (init ${OFFSET_INIT}), ${ITERS} steps -> ${dst}"
    if ! $PYTHON tools/optimize_ada_offset.py --mode optimize \
        --pn 1M --model_type infinity_2b --model_path weights/infinity_2b_reg.pth --vae_type 32 --vae_path weights/infinity_vae_d32reg.pth \
        --add_lvl_embeding_only_first_block 1 --use_bit_label 1 --rope2d_each_sa_layer 1 --rope2d_normalized_by_hw 2 \
        --use_scale_schedule_embedding 0 --apply_spatial_patchify 0 --checkpoint_type torch --bf16 1 --text_channels 2048 \
        --seed "$SEED" --images "${VICO_DIR}/${subject}+photo/imgs" --hflip 1 --grad_checkpointing 1 --offset_mode per_scale \
        --offset_init "$OFFSET_INIT" --init_std "$INIT_STD" --null_len 32 \
        --iters "$ITERS" --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" --lr "$LR" --save_offset_every "${SAVE_EVERY:-0}" --loss_scales "$LOSS_SCALES" \
        --log_every 10 --save_offset "${dst}/offset.pth" --save_file '' > "${dst}/optimize.log" 2>&1; then
        echo "   FAILED, see ${dst}/optimize.log"; tail -n 5 "${dst}/optimize.log"; failed+=("$subject"); continue
    fi
    grep -E '^\[it ' "${dst}/optimize.log" | tail -n 1
done
[ ${#failed[@]} -eq 0 ] || { echo "failed: ${failed[*]}"; exit 1; }
