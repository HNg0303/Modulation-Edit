# Vast.ai setup for Modulation-Edit

This repository contains the original `module_space` AdaLN-offset implementation, the reference images under `contents/`, and the style references under `styles/`. Model checkpoints are intentionally not stored in GitHub; download them into `weights/` on the GPU machine.

## Use the friend-synchronized branch

This branch contains the friend-synchronized code, the project data, and the
Vast.ai setup scripts. If the repository is not on the server yet:

```bash
cd /workspace
git clone -b friend_module_space_sync https://github.com/LeeHoang2710/Modulation-Edit.git
cd Modulation-Edit
```

If the repository is already there, update it without overwriting local work:

```bash
cd /workspace/Modulation-Edit
git fetch origin
git switch friend_module_space_sync
git pull --ff-only origin friend_module_space_sync
```

## 1. Create the Vast.ai machine

Use a CUDA-enabled Ubuntu image with a high-VRAM GPU. The 1M experiment below
uses a **five-image micro-batch**, which can require substantially more VRAM
than the earlier batch-size-2 run. It has not yet been measured on the new
machine; 32–40 GB is not guaranteed to be sufficient.

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

## 6. Run the dog scale-range experiment

With the conda environment activated, run from the repository root:

```bash
cd /workspace/Modulation-Edit
conda activate infinity-modulation
bash scripts/run_dog_prompt_modulation_experiment.sh
```

The script reads `contents/dog` (or `data/dreambooth/dataset/dog` if the first
folder is absent). It runs **four independent optimizations** on scales
`0-3`, `0-4`, `0-5`, and `0-6`. Each run uses the same seed, 50 optimizer
steps, `batch_size=5`, `grad_accum=1`, BF16, and horizontal flips. With five
original dog images, the batch contains five randomly selected examples from
the originals **and their flips**; it is not guaranteed to contain each
original image once at every step. The loss and learned offset are restricted
to that run's scale range.

After each optimization, inference keeps `cfg_uncond` as the base condition,
adds the learned offset only on its trained scales, and injects the new prompt
through cross-attention **and** `alpha * (G_prompt - G_null)` on every scale.
The five default prompts contain no word "dog": sleeping on a bed, running
through snow, wearing a red hat, seen from behind on a beach, and jumping over
a fallen log. For each scale range it tests alpha `0,0.25,0.5,1`, CFG `1,1.5`,
and seeds `0..4`: **200 generated images per range, 800 total**. CFG `1` uses
only the conditional row; CFG `1.5` also evaluates the null cross-attention
row with the same subject offset.

Results are stored separately:

```text
output/ada_offset/dog_prompt_modulation_50steps_bs5/
├── scales_0-3/
│   ├── offset.pth
│   ├── optimize.log
│   ├── optimization_reconstruction.jpg
│   └── inference/dog/<prompt>/grid.jpg
├── scales_0-4/
├── scales_0-5/
└── scales_0-6/
```

Individual generated images and `infer.log` are also saved under each
`inference/` directory. For a quick single-range test before launching all
four runs:

```bash
SCALE_ENDS=4 bash scripts/run_dog_prompt_modulation_experiment.sh
```

To change the data or output location:

```bash
IMAGES_DIR=contents/dog EXPERIMENT_DIR=output/my_dog_test \
  bash scripts/run_dog_prompt_modulation_experiment.sh
```

The script refuses to overwrite an existing offset. To repeat an experiment,
choose a new `EXPERIMENT_DIR`. If a run stops after saving its offset, rerun
inference only, for example:

```bash
OFFSET_PATH=output/ada_offset/dog_prompt_modulation_50steps_bs5/scales_0-3/offset.pth \
PROMPTS='sleeping on a bed|running through snow|wearing a red hat|seen from behind on a beach|jumping over a fallen log' \
ALPHAS=0,0.25,0.5,1 CFG=1,1.5 NUM_SAMPLES=5 VARIANTS=combined \
OUT_DIR=output/ada_offset/dog_prompt_modulation_50steps_bs5/scales_0-3/inference \
  bash scripts/infer_offset_prompt_modulation.sh dog
```

## 7. Common checks

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

If the five-image micro-batch runs out of GPU memory, use a higher-VRAM
instance. `PN=0.25M` is a lower-resolution diagnostic, not an equivalent 1M
comparison. Keep model weights in `weights/`; do not commit them to GitHub.
