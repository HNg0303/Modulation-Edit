"""Test a stronger prompt blend at early scales and a weaker blend at later scales.

Scale 0-3: alpha = [0.20, 0.15, 0.10, 0.05]
Scale 0-4: alpha = [0.25, 0.20, 0.15, 0.10, 0.05]

Conditional G_s = (1-alpha_s) * (G_null + D_s) + alpha_s * G_prompt.
The CFG partner remains G_null + D_s with null cross-attention. After the
learned scales, modulation is G_null, while conditional cross-attention uses
the full text prompt at every scale. No new offsets are trained here.
"""

import argparse
import json
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
CFGS = (1.0, 1.5)


def blend_coefficients(end_scale):
    """Only the conditional row is changed; the CFG null row keeps D_s."""
    return {si: (1.0 - alpha, alpha)
            for si, alpha in enumerate(SCHEDULES[end_scale])}


def make_comparison(prompt_dir):
    """Two CFG rows x five independently trained offset-seed columns."""
    tile, header = 512, 66
    canvas = Image.new("RGB", (5 * tile, 2 * (tile + header)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for row, cfg in enumerate(CFGS):
        for seed in range(5):
            source = prompt_dir / f"seed{seed}" / f"cfg{cfg:g}.png"
            if not source.is_file():
                raise FileNotFoundError(source)
            with Image.open(source) as image:
                thumb = ImageOps.contain(image.convert("RGB"), (tile, tile))
            x, y = seed * tile, row * (tile + header)
            draw.rectangle((x, y, x + tile - 1, y + header - 1), fill=(245, 245, 245))
            draw.text((x + 12, y + 9), f"offset train seed {seed}", fill=(20, 20, 20), font=font)
            draw.text((x + 12, y + 37), f"scale-wise alpha, CFG={cfg:g}",
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
    parser.add_argument("--top_k", type=int, default=900)
    parser.add_argument("--top_p", type=float, default=0.97)
    parser.add_argument("--skip_existing", action="store_true")
    parser.set_defaults(h_div_w_template=-1.)
    args = parser.parse_args()

    if not (args.use_bit_label and args.vae_type != 0 and not args.apply_spatial_patchify):
        parser.error("Requires bitwise, non-patchified Infinity-2B settings")
    if args.infer_seed < 0:
        parser.error("infer_seed must be nonnegative")
    for end_scale in SCHEDULES:
        for seed in range(5):
            path = Path(args.offset_root) / f"scales_0-{end_scale}" / f"train_seed{seed}" / "offset.pth"
            if not path.is_file():
                parser.error(f"Missing checkpoint: {path}")

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
        coefficients = blend_coefficients(end_scale)
        if offset_scales != set(coefficients) or len(schedule) <= end_scale:
            raise ValueError(f"Offset/schedule mismatch for scales 0-{end_scale}")
        scale_dir = Path(args.output_dir) / f"scale_0-{end_scale}"
        scale_dir.mkdir(parents=True, exist_ok=True)
        (scale_dir / "alpha_schedule.json").write_text(
            json.dumps({"alpha": list(SCHEDULES[end_scale]), "cfg": list(CFGS),
                        "infer_seed": args.infer_seed}, indent=2) + "\n", encoding="utf-8")
        print(f"[schedule] scales=0-{end_scale} alpha={SCHEDULES[end_scale]}; "
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
                for cfg in CFGS:
                    image_path = seed_dir / f"cfg{cfg:g}.png"
                    if args.skip_existing and image_path.is_file():
                        print(f"[skip] {image_path}", flush=True)
                        continue
                    conditional = (NULL, PROMPT, 1.0, 0.0)
                    unconditional = (NULL, NULL, 1.0, 0.0)
                    rows = [conditional] if cfg == 1.0 else [conditional, unconditional]
                    image = modspace_infer(
                        infinity, vae, text_cache[null_len], offset, rows,
                        schedule, offset_scales, offset_scales, cfg, 1.0,
                        args.infer_seed, args, scale_coefficients=coefficients,
                    )
                    if not cv2.imwrite(str(image_path), image):
                        raise OSError(f"Could not save {image_path}")
                    print(f"[saved] {image_path}", flush=True)
            print(f"[comparison] {make_comparison(prompt_dir)}", flush=True)


if __name__ == "__main__":
    main()
