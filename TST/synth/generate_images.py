"""
Generate JPEG images named image0.jpg, image1.jpg, ... with (approximately) linearly increasing
pixel areas between a lower and upper bound. Aspect ratios are random within user-specified bounds.

Notes & Constraints:
- Standard JPEG encoders typically cap width/height to ~65,535 pixels per side.
- Huge images will be extremely large on disk and slow to write; use --count to control how many.
- If a target size would exceed the per-side cap, this script scales it down to the max side length,
  preserving the chosen aspect ratio (so the realized area may be smaller than the target).
- Areas are spaced linearly between --min_area and --max_area (inclusive).
- By default, we generate 50 images spanning 1,000 to 10^10 pixels (the upper bound will be clipped
  as needed to fit JPEG side limits).

Example:
    python generate_images.py --count 25 --min_area 1000 --max_area 1e10 --min_ar 0.5 --max_ar 2.0 --outdir ./out
"""

import argparse
import math
import os
import random
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont

# Conservative JPEG dimension cap (many encoders/decoders use 65535 max per side)
JPEG_MAX_SIDE = 65535

def clip_max_area_for_jpeg(max_area: float) -> int:
    """Clip max_area to what is reachable with JPEG side limits."""
    max_reachable = JPEG_MAX_SIDE * JPEG_MAX_SIDE
    return int(min(max_area, max_reachable))

def choose_aspect_ratio(min_ar: float, max_ar: float) -> float:
    """Return a random aspect ratio r = width/height within [min_ar, max_ar]."""
    if min_ar <= 0 or max_ar <= 0:
        raise ValueError("Aspect ratio bounds must be positive.")
    if min_ar > max_ar:
        min_ar, max_ar = max_ar, min_ar
    return random.uniform(min_ar, max_ar)

def dims_from_area_and_ar(area: int, ar: float) -> Tuple[int, int]:
    """Compute integer (width, height) for given area and aspect ratio."""
    w_float = math.sqrt(area * ar)
    h_float = w_float / ar
    w = max(1, int(round(w_float)))
    h = max(1, int(round(h_float)))
    return w, h

def fit_to_jpeg_limits(w: int, h: int) -> Tuple[int, int, float]:
    """Scale dimensions down if they exceed JPEG_MAX_SIDE."""
    scale = 1.0
    if w > JPEG_MAX_SIDE or h > JPEG_MAX_SIDE:
        scale = min(JPEG_MAX_SIDE / float(w), JPEG_MAX_SIDE / float(h))
        w = max(1, int(math.floor(w * scale)))
        h = max(1, int(math.floor(h * scale)))
    return w, h, scale

def area_sequence_linear(min_area: int, max_area: int, count: int):
    """Generate linearly spaced areas."""
    if count <= 0:
        return []
    if count == 1:
        return [int(round((min_area + max_area) / 2.0))]
    step = (max_area - min_area) / float(count - 1)
    return [int(round(min_area + i * step)) for i in range(count)]

def draw_label(img: Image.Image, text: str) -> None:
    """Overlay a small label with dimensions and area."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    margin = 5
    draw.rectangle([0, 0, 420, 30], fill=(0, 0, 0))
    draw.text((margin, margin), text, fill=(255, 255, 255), font=font)

def main():
    parser = argparse.ArgumentParser(description="Generate JPEGs with linearly increasing pixel area.")
    parser.add_argument("--count", type=int, default=50, help="How many images to generate (default: 50).")
    parser.add_argument("--min_area", type=float, default=1000.0, help="Minimum pixel area (default: 1000).")
    parser.add_argument("--max_area", type=float, default=1e10, help="Maximum pixel area (default: 1e10; clipped).")
    parser.add_argument("--min_ar", type=float, default=0.5, help="Minimum aspect ratio (w/h) (default: 0.5).")
    parser.add_argument("--max_ar", type=float, default=2.0, help="Maximum aspect ratio (w/h) (default: 2.0).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: None).")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory (default: .).")
    parser.add_argument("--prefix", type=str, default="image", help="Filename prefix (default: image).")
    parser.add_argument("--start_index", type=int, default=0, help="Starting index for filenames (default: 0).")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    os.makedirs(args.outdir, exist_ok=True)

    min_area = max(1, int(math.floor(args.min_area)))
    max_area_user = int(math.floor(args.max_area))
    max_area = clip_max_area_for_jpeg(max_area_user)
    if max_area < max_area_user:
        print(f"[warn] Clipping max_area from {max_area_user} to {max_area} due to JPEG side limits.")

    if min_area > max_area:
        raise ValueError(f"min_area ({min_area}) cannot exceed max_area ({max_area}).")

    areas = area_sequence_linear(min_area, max_area, args.count)

    for i, target_area in enumerate(areas, start=args.start_index):
        ar = choose_aspect_ratio(args.min_ar, args.max_ar)
        w, h = dims_from_area_and_ar(target_area, ar)
        w, h, scale = fit_to_jpeg_limits(w, h)
        realized_area = w * h

        # Fill with random color
        color = tuple(random.randint(0, 255) for _ in range(3))
        img = Image.new("RGB", (w, h), color=color)

        # Overlay label
        label = f"{w}x{h}  area={realized_area:,}  target≈{target_area:,}  ar≈{ar:.3f}  scale={scale:.4f}"
        try:
            draw_label(img, label)
        except Exception:
            pass

        fname = os.path.join(args.outdir, f"{args.prefix}{i}.jpg")
        img.save(fname, format="JPEG", quality=85, optimize=True, progressive=True)
        print(f"Wrote {fname}: {label}")

if __name__ == "__main__":
    main()