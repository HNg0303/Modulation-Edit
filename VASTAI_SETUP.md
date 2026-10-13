# Vast.ai setup for Modulation-Edit

This repository contains the original `module_space` AdaLN-offset implementation, the reference images under `contents/`, and the style references under `styles/`. Model checkpoints are intentionally not stored in GitHub; download them into `weights/` on the GPU machine.

## 1. Create the Vast.ai machine

Use a CUDA-enabled Ubuntu image with at least 32 GB of VRAM for the 1M / 1024px experiment. Open a terminal in the repository after connecting:

```bash
cd /workspace/Modulation-Edit
git switch module_space
```

If the repository is not present yet:

```bash
git clone -b module_space https://github.com/LeeHoang2710/Modulation-Edit.git
cd Modulation-Edit
```

## 2. Install Miniconda

If `conda` is not available:

```bash
wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash miniconda.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda init bash
```

Start a new shell or source the conda script again, then verify:

```bash
conda --version
```

## 3. Create the environment

The original model imports FlashAttention directly. The setup script installs Python 3.10, CUDA PyTorch 2.5.1, the matching prebuilt FlashAttention wheel, and the remaining requirements:

```bash
bash scripts/setup_vastai_env.sh
conda activate infinity-modulation
```

The prebuilt FlashAttention wheel targets CUDA 12 and PyTorch 2.5.1. If the machine uses a different CUDA/PyTorch combination, set `TORCH_INDEX_URL` and `FLASH_ATTENTION_WHEEL` before running the script, or use a compatible Vast.ai image.

## 4. Download the model weights

```bash
bash scripts/download_vastai_weights.sh
```

This downloads and verifies:

```text
weights/infinity_2b_reg.pth
weights/infinity_vae_d32reg.pth
weights/flan-t5-xl/
```

If the Hugging Face repository requires authentication:

```bash
export HF_TOKEN=your_huggingface_token
bash scripts/download_vastai_weights.sh
```

The script is resumable and skips files that are already complete.

## 5. Verify the reference data

The repository already contains the copied datasets:

```text
contents/cat/00.jpg
contents/dog/00.jpg
contents/backpack_dog/00.jpg
styles/1.png
styles/2.png
...
```

To use another subject, place its images in a new folder under `contents/`. The original optimizer accepts JPG, JPEG, PNG, WebP, and BMP files.

## 6. Run the original AdaLN-offset experiment

The prepared launcher uses the original friend implementation with the current experiment settings:

- one image per optimizer step;
- gradient accumulation of 1;
- 400 optimizer steps by default;
- per-scale AdaLN offsets;
- horizontal-flip augmentation;
- cached VAE features and Bitwise Self-Correction;
- null-text (`cfg_uncond`) conditioning.

Run it with:

```bash
bash scripts/run_contents_offset.sh dog
```

For the hybrid experiment—learn the first four scales with null conditioning,
then use a text prompt for later scales—run:

```bash
bash scripts/run_hybrid_contents_offset.sh dog
```

The default hybrid settings are:

```text
learned scales: 0-3
text-conditioned scales: 4-12
later prompt: "is sleeping"
CFG after the switch: 3
```

You can change the prompt or boundary:

```bash
HYBRID_PROMPT="is sleeping on a beach" \\
HYBRID_START_SCALE=4 \\
bash scripts/run_hybrid_contents_offset.sh dog
```

For another subject:

```bash
bash scripts/run_contents_offset.sh cat
```

The result is saved under:

```text
output/ada_offset/<subject>_lsall/
├── offset.pth
├── optimize.log
├── samples_vs_targets.jpg
└── samples_vs_targets_r0_*.jpg
```

## 7. Change experiment settings

The launcher accepts environment-variable overrides:

```bash
ITERS=800 LR=1e-2 PN=0.25M bash scripts/run_contents_offset.sh dog
```

For the first five scales only:

```bash
LOSS_SCALES=0-4 bash scripts/run_contents_offset.sh dog
```

To use a different output directory:

```bash
OUT_DIR=output/ada_offset/my_dog bash scripts/run_contents_offset.sh dog
```

## 8. Run the original launcher directly

The original friend launcher still expects the DreamBooth-style path `data/dreambooth/dataset/<subject>`. For the copied data and the prepared defaults, use `scripts/run_contents_offset.sh`; it calls the same original `tools/optimize_ada_offset.py` implementation but points it to `contents/<subject>`.

## 9. Common checks

Check the GPU and FlashAttention installation:

```bash
python - <<'PY'
import torch
import flash_attn
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
print(flash_attn.__version__)
PY
```

If CUDA memory is insufficient, first use `PN=0.25M`. Keep the model weights in `weights/` and never commit that directory to GitHub.
