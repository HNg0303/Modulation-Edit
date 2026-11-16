#!/bin/bash
# E0 (Fig. 11): is G semantic? SOS + cross-attn fixed to a neutral prompt ("a photo"); only G = shared_ada_lin(cond)
# is set to G_c1 / G_c2 (all scales) or interpolated G(t) on the first k scales. Plus controls:
# class prompt with G pinned to neutral, and amplified class direction G_n + a*(G_c - G_n).
# usage: bash scripts/run_g_swap_class.sh      env: PYTHON SEEDS EARLY TS ALPHAS C1 C2 OUT
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}
OUT=${OUT:-output/g_swap/e0}
EARLY=${EARLY:-4,5,6,13}
$PYTHON experiments/g_swap/g_swap.py --c1 "${C1:-cat}" --c2 "${C2:-dog}" --seeds "${SEEDS:-0,1,2,3}" \
    --early "$EARLY" --ts "${TS:-0,0.25,0.5,0.75,1}" --alphas "${ALPHAS:-5,20}" --out "$OUT"
for k in ${EARLY//,/ }; do $PYTHON experiments/g_swap/make_fig.py "$OUT" "$k"; done
echo "[done] $OUT/fig11_early*.jpg, fig_controls.jpg, fig_S_sweep_seed*.jpg"
