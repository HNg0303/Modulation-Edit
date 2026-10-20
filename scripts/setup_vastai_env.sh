#!/usr/bin/env bash
set -euo pipefail

# Create a clean Miniconda environment for the original Modulation-Edit
# AdaLN-offset experiments on Vast.ai.
# Usage:
#   bash scripts/setup_vastai_env.sh
#   INFINITY_CONDA_ENV=my-infinity bash scripts/setup_vastai_env.sh

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${INFINITY_CONDA_ENV:-infinity-modulation}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
FLASH_ATTENTION_MAX_JOBS="${FLASH_ATTENTION_MAX_JOBS:-2}"
FLASH_ATTENTION_WHEEL="${FLASH_ATTENTION_WHEEL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found. Install Miniconda in the Vast.ai instance first." >&2
    exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    echo "Using existing conda environment: ${ENV_NAME}"
else
    echo "Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

echo "Installing PyTorch ${TORCH_VERSION} and torchvision ${TORCHVISION_VERSION}"
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip setuptools wheel
conda run -n "${ENV_NAME}" python -m pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"

# The original Modulation-Edit model imports FlashAttention directly, so it is
# required here. Prefer the prebuilt wheel to avoid a long source compilation.
echo "Installing FlashAttention"
if ! conda run -n "${ENV_NAME}" python -m pip install "${FLASH_ATTENTION_WHEEL}"; then
    echo "Prebuilt FlashAttention wheel failed; trying a source build instead."
    conda run -n "${ENV_NAME}" env MAX_JOBS="${FLASH_ATTENTION_MAX_JOBS}" \
        python -m pip install flash-attn --no-build-isolation
fi

echo "Installing project requirements"
conda run -n "${ENV_NAME}" python -m pip install -r "${REPO_DIR}/requirements.txt"
conda run -n "${ENV_NAME}" python -m pip install \
    "huggingface_hub" \
    "ninja" \
    "packaging" \
    "psutil" \
    "opencv-fixer==0.2.5"

echo "Checking the environment"
conda run -n "${ENV_NAME}" python - <<'PY'
import torch
import transformers
import flash_attn

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("flash_attn:", flash_attn.__version__)
PY

echo
echo "Environment setup completed. Activate it with:"
echo "  conda activate ${ENV_NAME}"
echo "Then download the model weights with:"
echo "  bash scripts/download_vastai_weights.sh"
