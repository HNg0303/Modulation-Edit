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
folder is absent). It runs **five independent optimizations per scale range**
on scales `0-3`, `0-4`, `0-5`, and `0-6`: 20 saved offsets in total. The five
training seeds are `0 1 2 3 4`; each run uses 50 optimizer steps,
`batch_size=5`, `grad_accum=1`, and BF16. The script requires exactly five
original images. Each step sends all five originals through one batched
forward/backward pass, averages their reconstruction losses, and updates the
shared offset once. Horizontal flips and bitwise self-correction noise are
disabled; preprocessing still resizes and center-crops images to the model's
resolution. The training seed changes only the order within this full batch,
so the five saved offsets may be nearly identical. Each `offset.pth` records
its seed. The loss and learned offset are restricted to that run's scale
range.

After each optimization, inference keeps `cfg_uncond` as the base condition
and adds the learned offset only on its trained scales. Two independent
switches control the `combined` inference variant:

| `PROMPT_CA_KV` | `PROMPT_MOD_DELTA` | Prompt conditioning |
| --- | --- | --- |
| `0` | `0` | Neither (offset-only control) |
| `1` | `0` | Cross-attention only |
| `0` | `1` | `alpha * (G_prompt - G_null)` in AdaLN only |
| `1` | `1` | Both (default) |

The null condition remains the source of SOS and the output head in all four
cases. The switches do not affect optimization or the explicit `mod_delta`,
`ca_prompt`, and `baseline` variants. When `PROMPT_MOD_DELTA=0`, alpha has no
effect, so inference generates only one result per prompt and CFG setting.

Each of the five prompts explicitly mentions the entity: "a dog sleeping on a
bed", "a dog running through snow", "a dog wearing a red hat", "a dog seen
from behind on a beach", and "a dog jumping over a fallen log". This tests
text-guided editing **with the class noun present**; it is no longer an
entity-free-prompt test. By default, **each learned offset** tests alpha
`0.1,0.2,0.3,0.4,0.5` and CFG `1,1.5`, using **one fixed inference seed (`0`)**:
50 generated images per offset, 250 per scale range, 1000 total. With
`PROMPT_MOD_DELTA=0`, there are 10 generated images per offset. The five seeds
apply to optimization, **not** to inference. CFG `1` uses only the conditional
row; CFG `1.5` also evaluates the null cross-attention row with the same
subject offset.

Results are stored separately:

```text
output/ada_offset/dog_prompt_modulation_50steps_bs5_train5/
├── scales_0-3/
│   ├── train_seed0/
│   │   ├── offset.pth
│   │   ├── optimize.log
│   │   ├── optimization_reconstruction.jpg
│   │   └── inference_ca1_mod1/dog/<prompt>/grid.jpg
│   ├── train_seed1/
│   ├── train_seed2/
│   ├── train_seed3/
│   └── train_seed4/
├── scales_0-4/
├── scales_0-5/
└── scales_0-6/
```

Individual generated images and `infer.log` are also saved under each
`inference_ca1_mod1/` directory by default. The two suffixes record the
switches. To test an existing offset without optimizing again, run:

```bash
PROMPT_CA_KV=0 PROMPT_MOD_DELTA=1 \
  OFFSET_PATH=output/ada_offset/dog_prompt_modulation_50steps_bs5_train5/scales_0-3/train_seed0/offset.pth \
  VARIANTS=combined PROMPTS='a dog sleeping on a bed' ALPHAS=0.1,0.2,0.3,0.4,0.5 CFG=1,1.5 \
  bash scripts/infer_offset_prompt_modulation.sh dog
```

The resulting `combined` images have `ca_kv = null` at all scales; only the
AdaLN delta carries the edit prompt. Set `PROMPT_CA_KV=1 PROMPT_MOD_DELTA=0`
to test cross-attention only with the same saved offset. For a quick
single-range test before launching all four scale ranges and all five training
seeds:

```bash
EXPERIMENT_DIR=output/ada_offset/dog_prompt_modulation_smoke \
SCALE_ENDS=4 TRAIN_SEEDS='0' bash scripts/run_dog_prompt_modulation_experiment.sh
```

Set `TRAIN_SEEDS='0 1 2 3 4'` to select training runs, or
`INFER_SEED=7` to change the single inference seed **for every offset**.

To change the data or output location:

```bash
IMAGES_DIR=contents/dog EXPERIMENT_DIR=output/my_dog_test \
  bash scripts/run_dog_prompt_modulation_experiment.sh
```

The script refuses to overwrite an existing offset. To repeat an experiment,
choose a new `EXPERIMENT_DIR`. If a run stops after saving its offset, rerun
inference only, for example:

```bash
OFFSET_PATH=output/ada_offset/dog_prompt_modulation_50steps_bs5_train5/scales_0-3/train_seed0/offset.pth \
PROMPTS='a dog sleeping on a bed|a dog running through snow|a dog wearing a red hat|a dog seen from behind on a beach|a dog jumping over a fallen log' \
ALPHAS=0.1,0.2,0.3,0.4,0.5 CFG=1,1.5 ITER=50 SEED=0 NUM_SAMPLES=1 VARIANTS=combined \
OUT_DIR=output/ada_offset/dog_prompt_modulation_50steps_bs5_train5/scales_0-3/train_seed0/inference_ca1_mod1 \
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
