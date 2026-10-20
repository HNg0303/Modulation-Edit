"""Direction 1: add the text delta only to AdaLN scale/shift vectors.

Reuses the existing dog offsets unchanged, including all six learned vectors.
The five columns are offset-training seeds; the inference seed is fixed.
"""

import argparse
import os.path as osp
import sys
from pathlib import Path

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import cv2
import torch

from tools.infer_offset_modspace import NULL, PROMPT, encode_texts, modspace_infer
from tools.infer_offset_prompt import slug
from tools.make_dog_prompt_comparison import DEFAULT_VARIANTS, make_comparison, variant_filename
from tools.optimize_ada_offset import get_scale_schedule, parse_scales
from tools.run_infinity import (
    add_common_arguments,
    encode_prompt,
    load_tokenizer,
    load_transformer,
    load_visual_tokenizer,
)


PROMPTS = (
    "a dog sleeping on a bed",
    "a dog running through snow",
    "a dog wearing a red hat",
    "a dog seen from behind on a beach",
    "a dog jumping over a fallen log",
)


def load_offsets(args, end_scale, channels, max_null_len):
    expected = set(range(end_scale + 1))
    schedule = None
    offsets = []
    for seed in range(5):
        path = Path(args.offset_root) / f"scales_0-{end_scale}" / f"train_seed{seed}" / "offset.pth"
        checkpoint = torch.load(path, map_location="cpu")
        for field, value in (("iters", args.expected_steps), ("seed", seed),
                             ("batch_size", 5), ("grad_accum", 1),
                             ("hflip", 0), ("offset_mode", "per_scale")):
            if checkpoint.get(field) != value:
                raise ValueError(f"{path}: {field}={checkpoint.get(field)!r}, expected {value!r}")
        if set(checkpoint.get("loss_scales", [])) != expected:
            raise ValueError(f"{path}: expected loss_scales={sorted(expected)}")
        if checkpoint.get("pn", args.pn) != args.pn:
            raise ValueError(f"{path}: pn does not match {args.pn}")
        _, current_schedule, _ = get_scale_schedule(args, checkpoint.get("h_div_w_template", 1.0))
        if schedule is None:
            schedule = current_schedule
        elif schedule != current_schedule:
            raise ValueError(f"{path}: scale schedule differs")
        offset = checkpoint["ada_offset"]
        if offset.shape != (len(schedule), 1, 6, channels) or not torch.isfinite(offset).all():
            raise ValueError(f"{path}: invalid offset shape or non-finite values")
        null_len = checkpoint.get("null_len", 32)
        if not 0 < null_len <= max_null_len:
            raise ValueError(f"{path}: invalid null_len={null_len}")
        offsets.append((seed, null_len, offset.cuda().float()))
    return schedule, expected, offsets


def mask_prompt_gammas(texts):
    """Only the prompt-minus-null delta is masked; D_s stays untouched."""
    sos, gss, ca = texts
    if gss.ndim != 4 or gss.shape[2] != 6:
        raise ValueError(f"Expected GSS shape [sources, 1, 6, C], got {tuple(gss.shape)}")
    masked = gss.clone()
    # GSS order: gamma1, gamma2, scale1, scale2, shift1, shift2.
    masked[PROMPT, :, :2, :] = masked[NULL, :, :2, :]
    return sos, masked, ca


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--offset_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--infer_seed", type=int, default=0)
    parser.add_argument("--expected_steps", type=int, default=50)
    parser.add_argument("--delta_scales", default="0-4",
                        help="Prompt-delta scales; 0-4 matches the stated five-scale experiment, 'all' matches the local sweep script")
    parser.add_argument("--top_k", type=int, default=900)
    parser.add_argument("--top_p", type=float, default=0.97)
    parser.add_argument("--skip_existing", action="store_true")
    parser.set_defaults(h_div_w_template=-1.)
    args = parser.parse_args()

    if not (args.use_bit_label and args.vae_type != 0 and not args.apply_spatial_patchify):
        parser.error("Requires bitwise, non-patchified Infinity-2B settings")
    if args.infer_seed < 0:
        parser.error("infer_seed must be nonnegative")
    for end_scale in (3, 4):
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

    for end_scale in (3, 4):
        schedule, offset_scales, offsets = load_offsets(args, end_scale, infinity.C, len(infinity.cfg_uncond))
        delta_scales = set(parse_scales(args.delta_scales, len(schedule)))
        print(f"[direction1] offsets=0-{end_scale}; prompt delta scales={sorted(delta_scales)}; "
              "prompt gamma delta=0; learned D_s gamma unchanged", flush=True)
        scale_dir = Path(args.output_dir) / f"scale_0-{end_scale}"
        for prompt in PROMPTS:
            prompt_dir = scale_dir / slug(prompt)
            prompt_dir.mkdir(parents=True, exist_ok=True)
            (prompt_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
            text_cache = {}
            for train_seed, null_len, offset in offsets:
                if null_len not in text_cache:
                    text_cache[null_len] = mask_prompt_gammas(
                        encode_texts(infinity, null_len, prompt_cache[prompt]))
                seed_dir = prompt_dir / f"seed{train_seed}"
                seed_dir.mkdir(exist_ok=True)
                for alpha, cfg in DEFAULT_VARIANTS:
                    image_path = seed_dir / variant_filename(alpha, cfg)
                    if args.skip_existing and image_path.is_file():
                        print(f"[skip] {image_path}", flush=True)
                        continue
                    cond = (NULL, PROMPT, True, alpha)
                    uncond = (NULL, NULL, True, 0.0)
                    rows = [cond] if cfg == 1.0 else [cond, uncond]
                    image = modspace_infer(
                        infinity, vae, text_cache[null_len], offset, rows, schedule,
                        offset_scales, delta_scales, cfg, 1.0, args.infer_seed, args,
                    )
                    if not cv2.imwrite(str(image_path), image):
                        raise OSError(f"Could not save {image_path}")
                    print(f"[saved] {image_path}", flush=True)
            print(f"[comparison] {make_comparison(prompt_dir, list(range(5)))}", flush=True)


if __name__ == "__main__":
    main()
