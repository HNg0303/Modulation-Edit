"""Make a four-row, five-column comparison of prompt-modulation samples.

The five columns are independently trained offset seeds. Every image in the
comparison uses the same inference seed, so column differences are attributable
to the learned offsets rather than different sampling noise.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_VARIANTS = ((0.1, 1.0), (0.1, 1.5), (0.2, 1.0), (0.2, 1.5))


def variant_filename(alpha, cfg):
    return f"alpha{alpha:g}_cfg{cfg:g}.png"


def _font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_comparison(prompt_dir, seeds, variants=DEFAULT_VARIANTS, tile_size=512):
    """Read seedN/alphaX_cfgY.png and save comparison.png in prompt_dir."""
    prompt_dir = Path(prompt_dir)
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("Exactly five distinct training seeds are required")
    if len(variants) != 4:
        raise ValueError("Expected four alpha/CFG combinations")

    header_h = 66
    canvas = Image.new("RGB", (5 * tile_size, 4 * (tile_size + header_h)), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font(23)

    for row, (alpha, cfg) in enumerate(variants):
        for col, seed in enumerate(seeds):
            path = prompt_dir / f"seed{seed}" / variant_filename(alpha, cfg)
            if not path.is_file():
                raise FileNotFoundError(f"Cannot build comparison; missing {path}")
            with Image.open(path) as image:
                thumbnail = ImageOps.contain(image.convert("RGB"), (tile_size, tile_size))
            x = col * tile_size
            y = row * (tile_size + header_h)
            draw.rectangle((x, y, x + tile_size - 1, y + header_h - 1), fill=(245, 245, 245))
            draw.text((x + 12, y + 7), f"offset train seed {seed}", fill=(20, 20, 20), font=font)
            draw.text((x + 12, y + 34), f"alpha={alpha:g}, CFG={cfg:g}", fill=(65, 65, 65), font=font)
            canvas.paste(thumbnail, (x + (tile_size - thumbnail.width) // 2,
                                     y + header_h + (tile_size - thumbnail.height) // 2))

    result = prompt_dir / "comparison.png"
    canvas.save(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt_dir", type=Path)
    parser.add_argument("--train_seeds", default="0,1,2,3,4")
    args = parser.parse_args()
    seeds = [int(value) for value in args.train_seeds.split(",")]
    print(make_comparison(args.prompt_dir, seeds))


if __name__ == "__main__":
    main()
