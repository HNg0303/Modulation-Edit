"""Paired late-modulation ablation over ten independently trained 0-4 offsets.

Both variants use prompt cross-attention at every scale. Scales 0-4 use the
same learned-offset blend and early CFG. On scales 5-12, compare null versus
prompt modulation with independently configurable late CFG strengths.
"""

import argparse
import json
import math
import os.path as osp
import sys
from pathlib import Path

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import cv2
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from tools.infer_dog_prompt_scale_schedule import SCHEDULES, blend_coefficients
from tools.infer_dog_prompt_scale_shift_only import PROMPTS
from tools.infer_offset_modspace import NULL, PROMPT, encode_texts, modspace_infer
from tools.infer_offset_prompt import slug
from tools.optimize_ada_offset import get_scale_schedule
from tools.run_infinity import (
    add_common_arguments, encode_prompt, load_tokenizer, load_transformer,
    load_visual_tokenizer,
)

VARIANTS = (
    ("late_null", None),
    ("late_prompt", 5),
)


def load_offset(path, args, train_seed, channels, max_null_len):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected = {"iters": args.expected_steps, "seed": train_seed,
                "batch_size": 5, "grad_accum": 1, "hflip": 0,
                "offset_mode": "per_scale"}
    for field, value in expected.items():
        if checkpoint.get(field) != value:
            raise ValueError(f"{path}: {field}={checkpoint.get(field)!r}; expected {value!r}")
    if set(checkpoint.get("loss_scales", [])) != set(range(5)):
        raise ValueError(f"{path}: expected loss_scales=[0,1,2,3,4]")
    if checkpoint.get("pn", args.pn) != args.pn:
        raise ValueError(f"{path}: pn differs from {args.pn}")
    _, schedule, _ = get_scale_schedule(args, checkpoint.get("h_div_w_template", 1.0))
    if len(schedule) != 13:
        raise ValueError(f"Expected 13 scales, got {len(schedule)}")
    offset = checkpoint["ada_offset"]
    if offset.shape != (len(schedule), 1, 6, channels) or not torch.isfinite(offset).all():
        raise ValueError(f"{path}: invalid offset shape or non-finite values")
    null_len = checkpoint.get("null_len", 32)
    if not 0 < null_len <= max_null_len:
        raise ValueError(f"{path}: invalid null_len={null_len}")
    return schedule, null_len, offset.cuda().float()


def make_comparison(prompt_dir, train_seeds, infer_seed, late_cfgs):
    """Two variant rows, ten independently trained offset columns."""
    tile, header = 320, 44
    canvas = Image.new("RGB", (len(train_seeds) * tile, 2 * (tile + header)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for row, (variant, _) in enumerate(VARIANTS):
        for col, train_seed in enumerate(train_seeds):
            source = prompt_dir / f"train_seed{train_seed}" / f"{variant}.png"
            if not source.is_file():
                raise FileNotFoundError(source)
            with Image.open(source) as image:
                thumb = ImageOps.contain(image.convert("RGB"), (tile, tile))
            x, y = col * tile, row * (tile + header)
            draw.rectangle((x, y, x + tile - 1, y + header - 1), fill=(245, 245, 245))
            draw.text((x + 8, y + 14),
                      f"{variant} CFG={late_cfgs[variant]:g} | offset {train_seed}, infer {infer_seed}",
                      fill="black", font=font)
            canvas.paste(thumb, (x + (tile - thumb.width) // 2,
                                 y + header + (tile - thumb.height) // 2))
    result = prompt_dir / "comparison.png"
    canvas.save(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--offset_root", required=True)
    parser.add_argument("--output_dir", default="output8")
    parser.add_argument("--infer_seed", type=int, default=0)
    parser.add_argument("--expected_steps", type=int, default=50)
    parser.add_argument("--early_cfg", type=float, default=1.1)
    parser.add_argument("--late_null_cfg", type=float, default=1.1)
    parser.add_argument("--late_prompt_cfg", type=float, default=2.0)
    parser.add_argument("--top_k", type=int, default=900)
    parser.add_argument("--top_p", type=float, default=0.97)
    parser.set_defaults(h_div_w_template=-1.)
    args = parser.parse_args()
    if not (args.use_bit_label and args.vae_type != 0 and not args.apply_spatial_patchify):
        parser.error("Requires bitwise, non-patchified Infinity-2B settings")
    if args.infer_seed < 0:
        parser.error("infer_seed must be nonnegative")
    if any(not math.isfinite(value) or value <= 0 for value in
           (args.early_cfg, args.late_null_cfg, args.late_prompt_cfg)):
        parser.error("CFG values must be positive and finite")
    late_cfgs = {"late_null": args.late_null_cfg,
                 "late_prompt": args.late_prompt_cfg}
    train_seeds = tuple(range(10))
    offset_paths = [Path(args.offset_root) / "scales_0-4" /
                    f"train_seed{seed}" / "offset.pth" for seed in train_seeds]
    for path in offset_paths:
        if not path.is_file():
            parser.error(f"Missing offset checkpoint: {path}")

    config = {
        "offset_checkpoints": [
            {"path": str(path.resolve()), "size": path.stat().st_size,
             "mtime_ns": path.stat().st_mtime_ns} for path in offset_paths],
        "train_seeds": list(train_seeds), "infer_seed": args.infer_seed,
        "alpha_0_4": list(SCHEDULES[4]), "early_cfg": args.early_cfg,
        "late_variants": {"late_null": ["G_null", args.late_null_cfg],
                          "late_prompt": ["G_prompt", args.late_prompt_cfg]},
        "cross_attention": "prompt in conditional, null in CFG partner, all scales",
        "pn": args.pn, "top_k": args.top_k, "top_p": args.top_p,
        "model_path": str(Path(args.model_path).resolve()),
        "vae_path": str(Path(args.vae_path).resolve()),
        "text_encoder_ckpt": str(Path(args.text_encoder_ckpt).resolve()),
        "prompts": list(PROMPTS),
    }
    cfg_tag = (f"e{args.early_cfg:g}_n{args.late_null_cfg:g}_p{args.late_prompt_cfg:g}"
               .replace(".", "p"))
    run_dir = (Path(args.output_dir) /
               f"scale_0-4_train_seeds0-9_infer_seed{args.infer_seed}_{cfg_tag}")
    manifest = run_dir / "run_config.json"
    if manifest.is_file() and json.loads(manifest.read_text(encoding="utf-8")) != config:
        parser.error(f"Existing run has different settings: {manifest}; choose a new output directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    vae = load_visual_tokenizer(args)
    vae.requires_grad_(False)
    infinity = load_transformer(vae, args)
    infinity.requires_grad_(False)
    infinity.eval()
    offsets = [load_offset(path, args, seed, infinity.C, len(infinity.cfg_uncond))
               for seed, path in zip(train_seeds, offset_paths)]
    schedule = offsets[0][0]
    if any(current_schedule != schedule for current_schedule, _, _ in offsets[1:]):
        raise ValueError("The ten offset checkpoints have different scale schedules")
    reference_offset = offsets[0][2]
    for train_seed, (_, _, offset) in zip(train_seeds, offsets):
        distance = (offset - reference_offset).float().norm().item()
        print(f"[offset] train_seed={train_seed} norm={offset.float().norm().item():.5f} "
              f"distance_from_seed0={distance:.5f}", flush=True)
        if train_seed and distance == 0:
            print(f"[warning] train_seed{train_seed} has exactly the same offset as train_seed0", flush=True)
    tokenizer, encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    offset_scales = set(range(5))
    conditional = (NULL, PROMPT, 1.0, 0.0)
    unconditional = (NULL, NULL, 1.0, 0.0)

    for prompt in PROMPTS:
        prompt_dir = run_dir / slug(prompt)
        prompt_dir.mkdir(parents=True, exist_ok=True)
        (prompt_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
        prompt_cond = encode_prompt(tokenizer, encoder, prompt)
        text_cache = {}
        for train_seed, (_, null_len, offset) in zip(train_seeds, offsets):
            if null_len not in text_cache:
                text_cache[null_len] = encode_texts(infinity, null_len, prompt_cond)
            texts = text_cache[null_len]
            if train_seed == 0:
                delta_norm = (texts[1][PROMPT] - texts[1][NULL]).float().norm().item()
                print(f"[prompt modulation] prompt={prompt!r} "
                      f"norm(G_prompt-G_null)={delta_norm:.5f}", flush=True)
            train_dir = prompt_dir / f"train_seed{train_seed}"
            train_dir.mkdir(exist_ok=True)
            images = {}
            for variant, late_prompt_from_scale in VARIANTS:
                coefficients = blend_coefficients(4, len(schedule), late_prompt_from_scale)
                delta_scales = set(coefficients)
                cfg_by_scale = {si: late_cfgs[variant] for si in range(5, len(schedule))}
                # Both variants need two rows from scale 0: early CFG=1.1,
                # and their KV caches must remain aligned at later scales.
                rows = [conditional, unconditional]
                image = modspace_infer(
                    infinity, vae, texts, offset, rows, schedule,
                    offset_scales, delta_scales, args.early_cfg, 1.0, args.infer_seed, args,
                    scale_coefficients=coefficients, cfg_by_scale=cfg_by_scale,
                )
                path = train_dir / f"{variant}.png"
                if not cv2.imwrite(str(path), image):
                    raise OSError(f"Could not save {path}")
                images[variant] = image
                print(f"[saved] {path}", flush=True)
            difference = abs(images[VARIANTS[0][0]].astype("float32") -
                             images[VARIANTS[1][0]].astype("float32")).mean()
            print(f"[pair] prompt={prompt!r} train_seed={train_seed} "
                  f"infer_seed={args.infer_seed} pixel_MAE={difference:.3f}", flush=True)
        print(f"[comparison] {make_comparison(prompt_dir, train_seeds, args.infer_seed, late_cfgs)}", flush=True)


if __name__ == "__main__":
    main()
