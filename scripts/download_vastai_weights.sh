#!/usr/bin/env bash
set -euo pipefail

# Download the exact checkpoints expected by the experiment scripts.
# Usage:
#   bash scripts/download_vastai_weights.sh
#   bash scripts/download_vastai_weights.sh /workspace/Infinity/weights

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="${1:-${REPO_DIR}/weights}"
ENV_NAME="${INFINITY_CONDA_ENV:-infinity-modulation}"
HF_HOME_DIR="${HF_HOME:-${REPO_DIR}/.cache/huggingface}"

# Set USE_ACTIVE_PYTHON=1 when this script is run after `conda activate`.
# This avoids conda-run environment forwarding issues on some Vast.ai images.
USE_ACTIVE_PYTHON="${USE_ACTIVE_PYTHON:-0}"

mkdir -p "${WEIGHTS_DIR}" "${HF_HOME_DIR}"

if [[ "${USE_ACTIVE_PYTHON}" == "1" ]]; then
    PYTHON_BIN="${PYTHON_BIN:-python}"
else
    if ! command -v conda >/dev/null 2>&1; then
        echo "conda was not found. Activate the project environment or run scripts/setup_vastai_env.sh first." >&2
        exit 1
    fi

    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"

    if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
        echo "Conda environment '${ENV_NAME}' was not found." >&2
        echo "Run: bash scripts/setup_vastai_env.sh" >&2
        exit 1
    fi
    PYTHON_BIN="conda run --no-capture-output -n ${ENV_NAME} python"
fi

if ! ${PYTHON_BIN} -c 'import huggingface_hub' >/dev/null 2>&1; then
    echo "huggingface_hub is missing in the selected Python environment." >&2
    echo "Run: python -m pip install -U huggingface_hub" >&2
    exit 1
fi

echo "Downloading Infinity-2B and VAE checkpoints into ${WEIGHTS_DIR}"
HF_HOME="${HF_HOME_DIR}" HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}" \
${PYTHON_BIN} - "${WEIGHTS_DIR}" <<'PY'
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

weights_dir = Path(sys.argv[1]).resolve()
weights_dir.mkdir(parents=True, exist_ok=True)
token = os.environ.get("HF_TOKEN") or None
repo_id = "FoundationVision/Infinity"

for filename in ("infinity_2b_reg.pth", "infinity_vae_d32reg.pth"):
    destination = weights_dir / filename
    if destination.exists() and destination.stat().st_size > 100 * 1024 * 1024:
        print(f"Already present: {destination}")
        continue
    print(f"Downloading {repo_id}/{filename}")
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(weights_dir),
        token=token,
    )

text_dir = weights_dir / "flan-t5-xl"
if (text_dir / "config.json").exists():
    print(f"Already present: {text_dir}")
else:
    print("Downloading google/flan-t5-xl")
    snapshot_download(
        repo_id="google/flan-t5-xl",
        local_dir=str(text_dir),
        token=token,
    )
PY

check_file() {
    local path="$1"
    local minimum_bytes="$2"
    if [[ ! -f "${path}" ]]; then
        echo "Missing: ${path}" >&2
        exit 1
    fi
    local size
    size="$(stat -c '%s' "${path}" 2>/dev/null || stat -f '%z' "${path}")"
    if (( size < minimum_bytes )); then
        echo "File is suspiciously small or incomplete: ${path} (${size} bytes)" >&2
        exit 1
    fi
    echo "Verified: ${path} (${size} bytes)"
}

check_file "${WEIGHTS_DIR}/infinity_2b_reg.pth" $((8 * 1024 * 1024 * 1024))
check_file "${WEIGHTS_DIR}/infinity_vae_d32reg.pth" $((1024 * 1024 * 1024))
check_file "${WEIGHTS_DIR}/flan-t5-xl/config.json" 100

echo
echo "Weights are ready:"
printf '  %s\n' \
    "${WEIGHTS_DIR}/infinity_2b_reg.pth" \
    "${WEIGHTS_DIR}/infinity_vae_d32reg.pth" \
    "${WEIGHTS_DIR}/flan-t5-xl"
