#!/bin/bash
# ARBooth / DreamMatcher metrics (DINO, CLIP-I, CLIP-T, CLIP-T-cls, DIV) of every variant folder of one ablation part,
# with CSD-VAR/evaluate_personalization.py (same code as the CSD-VAR baseline), then one summary table.
# usage: bash ablation/eval_vico.sh <part_dir>      e.g. output/ablation_vico/02_blend
set -uo pipefail
source "$(dirname "$0")/_env.sh"
part=${1:?part dir}
data_dir=$VICO_DIR
if [ "$NUM_PROMPTS" != 0 ]; then  # subset run: evaluate the first NUM_PROMPTS prompts only
    data_dir=${part}/_vico_p${NUM_PROMPTS}
    mkdir -p "$data_dir"
    cp "$VICO_DIR/subjects.csv" "$data_dir/"
    for f in "$VICO_DIR"/prompts_*.txt; do head -n "$NUM_PROMPTS" "$f" > "$data_dir/$(basename "$f")"; done
    for d in "$VICO_DIR"/*+photo; do ln -sfn "$d" "$data_dir/$(basename "$d")"; done
fi
for vdir in "$part"/*/; do
    vdir=${vdir%/}
    [ -f "$vdir/config.json" ] || continue
    if [ -f "$vdir/metrics.csv" ] && [ "$vdir/metrics.csv" -nt "$vdir/config.json" ] && [ "${FORCE_EVAL:-0}" != 1 ]; then
        echo "== skip eval $(basename "$vdir"): metrics.csv up to date"; continue
    fi
    echo "== eval $(basename "$vdir")"
    $EVAL_PYTHON /workspace/CSD-VAR/evaluate_personalization.py --result_dir "$vdir" --data_dir "$data_dir" \
        --arbooth_root "$ARBOOTH_ROOT" > "$vdir/eval.log" 2>&1 || { echo "   FAILED, see $vdir/eval.log"; tail -n 5 "$vdir/eval.log"; }
    grep '^\[overall\]' "$vdir/eval.log" || true
done
$PYTHON ablation/summarize.py "$part"
