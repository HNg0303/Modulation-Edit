"""Test a stronger prompt blend at early scales and a weaker blend at later scales.

Scale 0-3: alpha = [0.20, 0.15, 0.10, 0.05]
Scale 0-4: alpha = [0.25, 0.20, 0.15, 0.10, 0.05]

Conditional G_s = (1-alpha_s) * (G_null + D_s) + alpha_s * G_prompt.
The CFG partner remains G_null + D_s with null cross-attention. By default,
modulation after the learned scales is G_null. With --late_prompt_from_scale 5,
the conditional modulation is G_prompt on scales 5-12; scale 4 stays G_null
for the 0-3 offset experiment. Optional --early_cfgs 1.1 --late_cfgs 2.0
sets CFG independently for scales 0-4 and 5-12 at the logits sampling step. Conditional
cross-attention uses the full text prompt at every scale. No new offsets are
trained here.
"""

import argparse
import hashlib
import json
import math
import os.path as osp
import sys
from pathlib import Path

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps

from tools.infer_dog_prompt_scale_shift_only import PROMPTS, load_offsets
from tools.infer_offset_modspace import NULL, PROMPT, encode_texts, modspace_infer
from tools.infer_offset_prompt import slug
from tools.run_infinity import (
    add_common_arguments, encode_prompt, load_tokenizer, load_transformer,
    load_visual_tokenizer,
)


SCHEDULES = {
    3: (0.20, 0.15, 0.10, 0.05),
    4: (0.25, 0.20, 0.15, 0.10, 0.05),
}


def cfg_filename(early_cfg, late_cfg=None):
    """Include both guidance values so changing either never reuses an image."""
    if late_cfg is None:
        late_cfg = early_cfg
    return f"cfg_early{early_cfg:g}_late{late_cfg:g}.png"


def run_identity(args, early_cfgs, late_cfgs):
    """Stable run directory for parameters that affect generated images."""
    config = {
        "early_cfgs": list(early_cfgs),
        "late_cfgs": list(late_cfgs) if late_cfgs is not None else None,
        "late_prompt_from_scale": args.late_prompt_from_scale,
        "alpha_schedules": {str(end): list(values) for end, values in SCHEDULES.items()},
        "infer_seed": args.infer_seed,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "pn": args.pn,
        "expected_steps": args.expected_steps,
        "offset_root": str(Path(args.offset_root).resolve()),
        "offset_checkpoints": [
            {"path": str(path.resolve()), "size": path.stat().st_size,
             "mtime_ns": path.stat().st_mtime_ns}
            for end_scale in SCHEDULES for seed in range(5)
            for path in [Path(args.offset_root) / f"scales_0-{end_scale}" /
                         f"train_seed{seed}" / "offset.pth"]
        ],
        "model_path": str(Path(args.model_path).resolve()),
        "vae_path": str(Path(args.vae_path).resolve()),
        "text_encoder_ckpt": str(Path(args.text_encoder_ckpt).resolve()),
        "prompts": list(PROMPTS),
    }
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    pairs = "__".join(f"e{early:g}_l{late:g}" for early, late in
                      zip(early_cfgs, late_cfgs if late_cfgs is not None else early_cfgs))
    return f"run_{pairs}_{signature}", config


def blend_coefficients(end_scale, total_scales=None, late_prompt_from_scale=None):
    """Conditional (D_s, G_prompt-G_null) weights; CFG null row keeps D_s."""
    coefficients = {si: (1.0 - alpha, alpha)
                    for si, alpha in enumerate(SCHEDULES[end_scale])}
    if late_prompt_from_scale is not None:
        if total_scales is None or not end_scale < late_prompt_from_scale < total_scales:
            raise ValueError("late prompt must start after the learned scales and before the final scale")
        coefficients.update({si: (0.0, 1.0)
                             for si in range(late_prompt_from_scale, total_scales)})
    return coefficients


def make_comparison(prompt_dir, early_cfgs, late_cfgs=None):
    """One row per CFG setting, five independently trained offset-seed columns."""
    tile, header = 512, 66
    canvas = Image.new("RGB", (5 * tile, len(early_cfgs) * (tile + header)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for row, cfg in enumerate(early_cfgs):
        for seed in range(5):
            source = prompt_dir / f"seed{seed}" / cfg_filename(
                cfg, late_cfgs[row] if late_cfgs is not None else None)
            if not source.is_file():
                raise FileNotFoundError(source)
            with Image.open(source) as image:
                thumb = ImageOps.contain(image.convert("RGB"), (tile, tile))
            x, y = seed * tile, row * (tile + header)
            draw.rectangle((x, y, x + tile - 1, y + header - 1), fill=(245, 245, 245))
            draw.text((x + 12, y + 9), f"offset train seed {seed}", fill=(20, 20, 20), font=font)
            cfg_label = (f"CFG 0-4={cfg:g}, 5-12={late_cfgs[row]:g}"
                         if late_cfgs is not None else f"scale-wise alpha, CFG={cfg:g}")
            draw.text((x + 12, y + 37), cfg_label,
                      fill=(65, 65, 65), font=font)
            canvas.paste(thumb, (x + (tile - thumb.width) // 2,
                                 y + header + (tile - thumb.height) // 2))
    result = prompt_dir / "comparison.png"
    canvas.save(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--offset_root", required=True)
    parser.add_argument("--output_dir", default="output2")
    parser.add_argument("--infer_seed", type=int, default=0)
    parser.add_argument("--expected_steps", type=int, default=50)
    parser.add_argument("--late_prompt_from_scale", type=int, default=None,
                        help="Use G_prompt for conditional modulation from this scale onward; "
                             "omit to preserve the original G_null behavior")
    parser.add_argument("--late_cfgs", default=None,
                        help="Late-scale CFG values paired with --early_cfgs; e.g. 2.0")
    parser.add_argument("--early_cfgs", default="1.0,1.5",
                        help="CFG values for scales 0-4; e.g. 1.1")
    parser.add_argument("--top_k", type=int, default=900)
    parser.add_argument("--top_p", type=float, default=0.97)
    parser.add_argument("--skip_existing", action="store_true")
    parser.set_defaults(h_div_w_template=-1.)
    args = parser.parse_args()

    if not (args.use_bit_label and args.vae_type != 0 and not args.apply_spatial_patchify):
        parser.error("Requires bitwise, non-patchified Infinity-2B settings")
    if args.infer_seed < 0:
        parser.error("infer_seed must be nonnegative")
    def parse_cfgs(spec, flag):
        try:
            values = tuple(float(value) for value in spec.split(","))
        except ValueError:
            parser.error(f"{flag} must contain numeric CFG values")
        if not 1 <= len(values) <= 2 or any(not math.isfinite(v) or v <= 0 for v in values):
            parser.error(f"{flag} must contain one or two positive, finite CFG values")
        return values

    early_cfgs = parse_cfgs(args.early_cfgs, "--early_cfgs")
    late_cfgs = parse_cfgs(args.late_cfgs, "--late_cfgs") if args.late_cfgs is not None else None
    if late_cfgs is not None and (len(late_cfgs) != len(early_cfgs) or args.late_prompt_from_scale != 5):
        parser.error("--late_cfgs must match --early_cfgs in length and requires --late_prompt_from_scale 5")
    for end_scale in SCHEDULES:
        for seed in range(5):
            path = Path(args.offset_root) / f"scales_0-{end_scale}" / f"train_seed{seed}" / "offset.pth"
            if not path.is_file():
                parser.error(f"Missing checkpoint: {path}")

    run_tag, run_config = run_identity(args, early_cfgs, late_cfgs)
    run_dir = Path(args.output_dir) / run_tag
    manifest = run_dir / "run_config.json"
    if manifest.is_file() and json.loads(manifest.read_text(encoding="utf-8")) != run_config:
        parser.error(f"Run configuration differs from existing {manifest}; choose a new output directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    print(f"[run] results={run_dir}; skip_existing={args.skip_existing}", flush=True)

    vae = load_visual_tokenizer(args)
    vae.requires_grad_(False)
    infinity = load_transformer(vae, args)
    infinity.requires_grad_(False)
    infinity.eval()
    tokenizer, encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    prompt_cache = {prompt: encode_prompt(tokenizer, encoder, prompt) for prompt in PROMPTS}

    for end_scale in SCHEDULES:
        schedule, offset_scales, offsets = load_offsets(
            args, end_scale, infinity.C, len(infinity.cfg_uncond))
        if args.late_prompt_from_scale is not None and args.late_prompt_from_scale != 5:
            parser.error("This comparison requires --late_prompt_from_scale 5")
        coefficients = blend_coefficients(
            end_scale, len(schedule), args.late_prompt_from_scale)
        if offset_scales != set(range(end_scale + 1)) or len(schedule) <= end_scale:
            raise ValueError(f"Offset/schedule mismatch for scales 0-{end_scale}")
        delta_scales = set(coefficients)
        scale_dir = run_dir / f"scale_0-{end_scale}"
        scale_dir.mkdir(parents=True, exist_ok=True)
        (scale_dir / "alpha_schedule.json").write_text(
            json.dumps({"alpha": list(SCHEDULES[end_scale]), "early_cfg": list(early_cfgs),
                        "late_cfg": list(late_cfgs) if late_cfgs is not None else None,
                        "infer_seed": args.infer_seed,
                        "late_prompt_from_scale": args.late_prompt_from_scale},
                       indent=2) + "\n", encoding="utf-8")
        print(f"[schedule] scales=0-{end_scale} alpha={SCHEDULES[end_scale]}; "
              f"late conditional modulation={'G_prompt from scale 5' if args.late_prompt_from_scale == 5 else 'G_null'}; "
              f"early CFG={early_cfgs}; late CFG={late_cfgs or early_cfgs}; "
              "CFG partner uses full D_s and null CA", flush=True)
        for prompt in PROMPTS:
            prompt_dir = scale_dir / slug(prompt)
            prompt_dir.mkdir(parents=True, exist_ok=True)
            (prompt_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
            text_cache = {}
            for train_seed, null_len, offset in offsets:
                if null_len not in text_cache:
                    text_cache[null_len] = encode_texts(
                        infinity, null_len, prompt_cache[prompt])
                seed_dir = prompt_dir / f"seed{train_seed}"
                seed_dir.mkdir(exist_ok=True)
                for row_idx, cfg in enumerate(early_cfgs):
                    image_path = seed_dir / cfg_filename(
                        cfg, late_cfgs[row_idx] if late_cfgs is not None else None)
                    if args.skip_existing and image_path.is_file():
                        print(f"[skip] {image_path}", flush=True)
                        continue
                    conditional = (NULL, PROMPT, 1.0, 0.0)
                    unconditional = (NULL, NULL, 1.0, 0.0)
                    # Keep both rows from scale 0 when late CFG is enabled,
                    # including the early CFG=1 case, to keep KV caches aligned.
                    rows = ([conditional, unconditional] if late_cfgs is not None or cfg != 1.0
                            else [conditional])
                    cfg_by_scale = ({si: late_cfgs[row_idx]
                                     for si in range(5, len(schedule))}
                                    if late_cfgs is not None else None)
                    image = modspace_infer(
                        infinity, vae, text_cache[null_len], offset, rows,
                        schedule, offset_scales, delta_scales, cfg, 1.0,
                        args.infer_seed, args, scale_coefficients=coefficients,
                        cfg_by_scale=cfg_by_scale,
                    )
                    if not cv2.imwrite(str(image_path), image):
                        raise OSError(f"Could not save {image_path}")
                    print(f"[saved] {image_path}", flush=True)
            print(f"[comparison] {make_comparison(prompt_dir, early_cfgs, late_cfgs)}", flush=True)


if __name__ == "__main__":
    main()
