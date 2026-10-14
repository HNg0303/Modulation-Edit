"""Compare class-relative prompt modulation against the existing combined baseline.

The only change is alpha * (G_full_prompt - G_"a dog") in place of
alpha * (G_full_prompt - G_null). Full-prompt cross-attention, null SOS/head,
all six learned offset vectors, sampling settings, and output grid are unchanged.
"""

import argparse
import os.path as osp
import sys
from pathlib import Path

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import cv2

from tools.infer_dog_prompt_scale_shift_only import PROMPTS, load_offsets
from tools.infer_offset_modspace import NULL, PROMPT, encode_texts, modspace_infer
from tools.infer_offset_prompt import slug
from tools.make_dog_prompt_comparison import DEFAULT_VARIANTS, make_comparison, variant_filename
from tools.optimize_ada_offset import parse_scales
from tools.run_infinity import (
    add_common_arguments,
    encode_prompt,
    load_tokenizer,
    load_transformer,
    load_visual_tokenizer,
)


def class_relative_texts(full_texts, class_texts):
    """Keep null base/CA intact; replace only the direction used by modspace_infer.

    modspace_infer computes delta = gss[PROMPT] - gss[NULL]. Replacing the
    PROMPT GSS by G_full - G_class + G_null makes delta exactly G_full-G_class.
    The prompt CA features and null SOS/head remain the original ones.
    """
    sos, gss, ca = full_texts
    class_gss = class_texts[1]
    if gss.shape != class_gss.shape or gss.ndim != 4 or gss.shape[2] != 6:
        raise ValueError("Full and class GSS must both have shape [3, 1, 6, C]")
    adjusted = gss.clone()
    adjusted[PROMPT] = gss[PROMPT] - class_gss[PROMPT] + gss[NULL]
    return sos, adjusted, ca


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--offset_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--infer_seed", type=int, default=0)
    parser.add_argument("--expected_steps", type=int, default=50)
    parser.add_argument("--class_prompt", default="a dog")
    parser.add_argument("--delta_scales", default="0-4",
                        help="Must match the comparison baseline; use 'all' for the local sweep script")
    parser.add_argument("--top_k", type=int, default=900)
    parser.add_argument("--top_p", type=float, default=0.97)
    parser.add_argument("--skip_existing", action="store_true")
    parser.set_defaults(h_div_w_template=-1.)
    args = parser.parse_args()

    if not (args.use_bit_label and args.vae_type != 0 and not args.apply_spatial_patchify):
        parser.error("Requires bitwise, non-patchified Infinity-2B settings")
    if args.infer_seed < 0:
        parser.error("infer_seed must be nonnegative")
    if args.class_prompt != "a dog":
        parser.error("This controlled dog experiment requires --class_prompt 'a dog'")
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
    prompt_cache = {prompt: encode_prompt(tokenizer, encoder, prompt)
                    for prompt in (*PROMPTS, args.class_prompt)}

    for end_scale in (3, 4):
        schedule, offset_scales, offsets = load_offsets(
            args, end_scale, infinity.C, len(infinity.cfg_uncond))
        delta_scales = set(parse_scales(args.delta_scales, len(schedule)))
        print(f"[class-delta] offsets=0-{end_scale}; delta scales={sorted(delta_scales)}; "
              "delta=G(full prompt)-G(a dog); CA=full prompt", flush=True)
        scale_dir = Path(args.output_dir) / f"scale_0-{end_scale}"
        class_text_cache = {}
        for prompt in PROMPTS:
            prompt_dir = scale_dir / slug(prompt)
            prompt_dir.mkdir(parents=True, exist_ok=True)
            (prompt_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
            text_cache = {}
            for train_seed, null_len, offset in offsets:
                if null_len not in text_cache:
                    if null_len not in class_text_cache:
                        class_text_cache[null_len] = encode_texts(
                            infinity, null_len, prompt_cache[args.class_prompt])
                    full_texts = encode_texts(infinity, null_len, prompt_cache[prompt])
                    text_cache[null_len] = class_relative_texts(
                        full_texts, class_text_cache[null_len])
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
