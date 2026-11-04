#!/bin/bash
# Reproduce output/ablation_vico/19_full_bsc (full ViCo: 16 subjects x 31 prompts x 8 seeds = 3968 images).
#   1) optimize: per-scale AdaLN offset D per subject, trained with the subject text (T5 '<name>'),
#      bitwise self-correction (noise_apply_layers 13, strength 0.3), NO augmentation (hflip 0),
#      200 it, bs 2 x grad_accum 2, lr 5e-3, loss on scales 0-4
#   2) generate: G_subj + 0.9 * D on scales 0,2-4, cross-attention prompt 'a {name} {condition}', cfg 1.5
#   3) eval: CSD-VAR evaluate_personalization.py (ARBooth protocol: DINO / CLIP-I / CLIP-T / DIV)
# usage: bash run_vico_bsc_best.sh [optimize|generate|eval|all]   (default all; finished subjects are skipped)
#        SUBJECTS=black_cat,red_vase bash run_vico_bsc_best.sh     (subset)
set -uo pipefail
cd "$(dirname "$0")"
STAGE=${1:-all}
source ablation/_env.sh >/dev/null
export PIP_CONFIG_FILE=/dev/null PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OFFSET_ROOT=${OFFSET_ROOT:-output/ablation_vico/offsets/ls0-4_subjtext_it200_bsc_noaug}
OUT=${OUT:-output/ablation_vico/19_full_bsc}
V='gsubj_bsc_s0,2-4_b0.9'

model=(--pn 1M --model_type infinity_2b --model_path weights/infinity_2b_reg.pth --vae_type 32 --vae_path weights/infinity_vae_d32reg.pth
       --add_lvl_embeding_only_first_block 1 --use_bit_label 1 --rope2d_each_sa_layer 1 --rope2d_normalized_by_hw 2
       --use_scale_schedule_embedding 0 --apply_spatial_patchify 0 --checkpoint_type torch --bf16 1 --text_channels 2048
       --text_encoder_ckpt weights/flan-t5-xl --seed 0 --null_len 32)

if [ -n "${SUBJECTS:-}" ]; then IFS=, read -ra SUBJ <<< "$SUBJECTS"
else SUBJ=($(tail -n +2 "$VICO_DIR/subjects.csv" | cut -d, -f2)); fi
echo "subjects (${#SUBJ[@]}): ${SUBJ[*]}"

# ---------------------------------------------------------------- 1) optimize
if [ "$STAGE" = optimize ] || [ "$STAGE" = all ]; then
  for s in "${SUBJ[@]}"; do
    OFF=$OFFSET_ROOT/$s
    [ -f "$OFF/offset.pth" ] && { echo "skip optimize $s"; continue; }
    mkdir -p "$OFF"
    name=$($PYTHON -c "import sys; sys.path.insert(0, 'ablation'); from common import subject_name; print(subject_name('$s'))" 2>/dev/null | tail -1)
    echo "[$(date)] optimize $s (train prompt '$name')"
    $PYTHON tools/optimize_ada_offset.py --mode optimize "${model[@]}" \
        --images "$VICO_DIR/$s+photo/imgs" --hflip 0 \
        --grad_checkpointing 1 --offset_mode per_scale \
        --iters 200 --batch_size 2 --grad_accum 2 --lr 5e-3 --loss_scales 0-4 \
        --noise_apply_layers 13 --noise_apply_strength 0.3 \
        --train_prompt "$name" \
        --log_every 10 --save_offset "$OFF/offset.pth" --save_file '' > "$OFF/optimize.log" 2>&1 \
      || { echo "FAILED optimize $s, see $OFF/optimize.log"; continue; }
    grep -E '^\[it  199' "$OFF/optimize.log" | cut -c1-60
  done
fi

# ---------------------------------------------------------------- 2) generate
if [ "$STAGE" = generate ] || [ "$STAGE" = all ]; then
  mkdir -p "$OUT/logs"
  for s in "${SUBJ[@]}"; do
    n=$(ls "$OUT/$V/$s"/*/ 2>/dev/null | grep -c jpg)
    [ "$n" -ge 248 ] && { echo "skip generate $s ($n imgs)"; continue; }
    echo "[$(date)] generate $s"
    SUBJECTS=$s NUM_PROMPTS=0 NUM_SAMPLES=8 source ablation/_env.sh >/dev/null   # all 31 prompts x 8 seeds
    .venv/bin/python - --out_dir "$OUT" --prompt_name subject "${infer_args[@]}" > "$OUT/logs/gen_$s.log" 2>&1 <<'PY' \
      || echo "FAILED generate $s, see $OUT/logs/gen_$s.log"
import argparse, sys
sys.path.insert(0, 'ablation')
from common import Variant, add_args, run
args = add_args(argparse.ArgumentParser()).parse_args()
run(args, [Variant(name='gsubj_bsc_s0,2-4_b0.9', alpha=0., ca='prompt',
                   offset_root='output/ablation_vico/offsets/ls0-4_subjtext_it200_bsc_noaug',
                   offset_scales='0,2-4', offset_weight=0.9, base_text='subject', cfg=1.5, tau=args.tau)])
PY
    echo "   $(ls "$OUT/$V/$s"/*/ 2>/dev/null | grep -c jpg) imgs"
  done
fi

# ---------------------------------------------------------------- 3) eval
if [ "$STAGE" = eval ] || [ "$STAGE" = all ]; then
  $EVAL_PYTHON /workspace/CSD-VAR/evaluate_personalization.py --result_dir "$OUT/$V" --data_dir "$VICO_DIR" \
      --arbooth_root "$ARBOOTH_ROOT" > "$OUT/$V/eval.log" 2>&1 || echo "FAILED eval, see $OUT/$V/eval.log"
  grep -E '^\[(overall|live|object)\]' "$OUT/$V/eval.log"
fi
echo "[$(date)] done ($STAGE)"
