"""Compare five dog offsets on five prompts with two alpha and CFG settings.

Output layout:
    <output_dir>/scale_0-3/<prompt>/seed0/alpha0.1_cfg1.png
    <output_dir>/scale_0-3/<prompt>/comparison.png
    <output_dir>/scale_0-4/...

The seedN directories refer to OFFSET TRAINING seeds. Sampling uses one fixed
--infer_seed for all columns, which makes comparisons fair.
"""

import argparse
import os
import os.path as osp
import sys
from pathlib import Path

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import cv2
import torch

from tools.infer_offset_modspace import NULL, PROMPT, encode_texts, modspace_infer
from tools.infer_offset_prompt import slug
from tools.make_dog_prompt_comparison import DEFAULT_VARIANTS, make_comparison, variant_filename
from tools.optimize_ada_offset import get_scale_schedule
from tools.run_infinity import (
    add_common_arguments,
    encode_prompt,
    load_tokenizer,
    load_transformer,
    load_visual_tokenizer,
)


DEFAULT_PROMPTS = (
    "a dog sleeping on a bed",
    "a dog running through snow",
    "a dog wearing a red hat",
    "a dog seen from behind on a beach",
    "a dog jumping over a fallen log",
)


def parse_csv(value, cast):
    return [cast(part.strip()) for part in value.split(",") if part.strip()]


def load_offsets(args, end_scale, model_channels):
    offsets = []
    expected_scales = set(range(end_scale + 1))
    schedule = None
    for seed in args.train_seeds:
        path = Path(args.offset_root) / f"scales_0-{end_scale}" / f"train_seed{seed}" / "offset.pth"
        if not path.is_file():
            raise FileNotFoundError(f"Missing offset: {path}")
        ckpt = torch.load(path, map_location="cpu")
        for field, expected in (("iters", args.expected_steps), ("seed", seed),
                                ("batch_size", 5), ("grad_accum", 1),
                                ("hflip", 0), ("offset_mode", "per_scale")):
            if ckpt.get(field) != expected:
                raise ValueError(f"{path}: {field}={ckpt.get(field)!r}; expected {expected!r}")
        trained_scales = set(ckpt.get("loss_scales", []))
        if trained_scales != expected_scales:
            raise ValueError(f"{path}: trained scales {sorted(trained_scales)}, expected {sorted(expected_scales)}")
        if ckpt.get("pn", args.pn) != args.pn:
            raise ValueError(f"{path}: checkpoint pn does not match {args.pn}")
        null_len = ckpt.get("null_len", 32)
        if null_len <= 0:
            raise ValueError(f"{path}: invalid null text length {null_len}")
        _, this_schedule, _ = get_scale_schedule(args, ckpt.get("h_div_w_template", 1.0))
        if schedule is None:
            schedule = this_schedule
        elif schedule != this_schedule:
            raise ValueError(f"{path}: scale schedule differs from other seeds")
        offset = ckpt["ada_offset"]
        if offset.shape != (len(schedule), 1, 6, model_channels):
            raise ValueError(f"{path}: unexpected offset shape {tuple(offset.shape)}")
        if not torch.isfinite(offset).all():
            raise ValueError(f"{path}: offset contains non-finite values")
        offsets.append((seed, null_len, offset.cuda().float()))
    return schedule, expected_scales, offsets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--offset_root", required=True,
                        help="Root with scales_0-3/train_seed0/offset.pth, etc.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scale_ends", default="3,4")
    parser.add_argument("--train_seeds", default="0,1,2,3,4")
    parser.add_argument("--infer_seed", type=int, default=0,
                        help="Held fixed across all training-seed columns")
    parser.add_argument("--expected_steps", type=int, default=50,
                        help="Reject reused offsets trained for a different number of steps")
    parser.add_argument("--prompts", default="|".join(DEFAULT_PROMPTS))
    parser.add_argument("--top_k", type=int, default=900)
    parser.add_argument("--top_p", type=float, default=0.97)
    parser.add_argument("--skip_existing", action="store_true")
    parser.set_defaults(h_div_w_template=-1.)
    args = parser.parse_args()

    args.scale_ends = parse_csv(args.scale_ends, int)
    args.train_seeds = parse_csv(args.train_seeds, int)
    prompts = [prompt.strip() for prompt in args.prompts.split("|") if prompt.strip()]
    if args.scale_ends != [3, 4]:
        parser.error("This focused sweep requires scale_ends=3,4")
    if len(args.train_seeds) != 5 or len(set(args.train_seeds)) != 5:
        parser.error("Provide exactly five distinct training seeds")
    if len(prompts) != 5 or len({slug(p) for p in prompts}) != 5:
        parser.error("Provide exactly five distinct prompts")
    if not (args.use_bit_label and args.vae_type != 0 and not args.apply_spatial_patchify):
        parser.error("Requires bitwise, non-patchified Infinity-2B settings")
    if args.infer_seed < 0:
        parser.error("infer_seed must be nonnegative")

    # Fail before loading the 2B model if a training checkpoint is missing.
    for end_scale in args.scale_ends:
        for seed in args.train_seeds:
            path = Path(args.offset_root) / f"scales_0-{end_scale}" / f"train_seed{seed}" / "offset.pth"
            if not path.is_file():
                parser.error(f"Missing checkpoint: {path}")

    vae = load_visual_tokenizer(args)
    vae.requires_grad_(False)
    infinity = load_transformer(vae, args)
    infinity.requires_grad_(False)
    infinity.eval()
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    prompt_cache = {prompt: encode_prompt(text_tokenizer, text_encoder, prompt) for prompt in prompts}

    for end_scale in args.scale_ends:
        schedule, offset_scales, offsets = load_offsets(args, end_scale, infinity.C)
        delta_scales = set(range(len(schedule)))  # alpha*(G_prompt-G_null) on ALL scales
        scale_dir = Path(args.output_dir) / f"scale_0-{end_scale}"

        for prompt in prompts:
            prompt_dir = scale_dir / slug(prompt)
            prompt_dir.mkdir(parents=True, exist_ok=True)
            (prompt_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
            texts_by_null_len = {}

            for train_seed, null_len, offset in offsets:
                if null_len not in texts_by_null_len:
                    texts_by_null_len[null_len] = encode_texts(infinity, null_len, prompt_cache[prompt])
                texts = texts_by_null_len[null_len]
                seed_dir = prompt_dir / f"seed{train_seed}"
                seed_dir.mkdir(exist_ok=True)

                for alpha, cfg in DEFAULT_VARIANTS:
                    image_path = seed_dir / variant_filename(alpha, cfg)
                    if args.skip_existing and image_path.is_file():
                        print(f"[skip] {image_path}", flush=True)
                        continue
                    cond = (NULL, PROMPT, True, alpha)  # null SOS/head, prompt CA, learned D_s + delta
                    uncond = (NULL, NULL, True, 0.0)  # same D_s, null CA, no delta
                    rows = [cond] if cfg == 1.0 else [cond, uncond]
                    image = modspace_infer(
                        infinity, vae, texts, offset, rows, schedule,
                        offset_scales, delta_scales, cfg, 1.0, args.infer_seed, args,
                    )
                    if not cv2.imwrite(str(image_path), image):
                        raise OSError(f"Could not save {image_path}")
                    print(f"[saved] {image_path}", flush=True)

            comparison = make_comparison(prompt_dir, args.train_seeds)
            print(f"[comparison] {comparison}", flush=True)


if __name__ == "__main__":
    main()
