# sourced by ablation/*.sh; every value can be overridden from the environment
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON=${PYTHON:-.venv/bin/python}                                   # Infinity env (torch 2.5.1)
EVAL_PYTHON=${EVAL_PYTHON:-/root/miniconda3/envs/csd-var/bin/python} # env with ARBooth booth_evaluation deps
VICO_DIR=${VICO_DIR:-/workspace/CSD-VAR/benchmark_data/vico_csd}
ARBOOTH_ROOT=${ARBOOTH_ROOT:-/workspace/ARBooth}
ROOT=${ROOT:-output/ablation_vico}
NUM_SAMPLES=${NUM_SAMPLES:-8}     # ViCo protocol: 8 images per prompt
NUM_PROMPTS=${NUM_PROMPTS:-0}     # 0 = all 31 prompts
BATCH=${BATCH:-1}
SUBJECTS=${SUBJECTS:-}            # comma-separated, empty = all 16
ALPHA=${ALPHA:-0.2}               # blend weight used by parts 1/3/4 (set it to part 2's best)
BLEND=${BLEND:-add}
CFG=${CFG:-1.5}
TAU=${TAU:-1.0}
SEED=${SEED:-0}
infer_args=(--num_samples "$NUM_SAMPLES" --num_prompts "$NUM_PROMPTS" --batch "$BATCH" --subjects "$SUBJECTS"
            --alpha "$ALPHA" --blend "$BLEND" --cfg "$CFG" --tau "$TAU" --seed "$SEED")
