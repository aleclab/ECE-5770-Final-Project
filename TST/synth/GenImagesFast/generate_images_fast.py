#!/usr/bin/env python3
import argparse
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple

from PIL import Image  # pip install pillow

JPEG_MAX_SIDE = 65535

def clip_max_area_for_jpeg(max_area: float) -> int:
    return int(min(max_area, JPEG_MAX_SIDE * JPEG_MAX_SIDE))

def choose_aspect_ratio(min_ar: float, max_ar: float) -> float:
    if min_ar <= 0 or max_ar <= 0:
        raise ValueError("Aspect ratio bounds must be positive.")
    if min_ar > max_ar:
        min_ar, max_ar = max_ar, min_ar
    return random.uniform(min_ar, max_ar)

def dims_from_area_and_ar(area: int, ar: float) -> Tuple[int, int]:
    # Continuous solution, then rounded to integers
    w = max(1, int(round(math.sqrt(area * ar))))
    h = max(1, int(round(w / ar)))
    if w * h == 0:
        w = max(1, w); h = max(1, h)
    return w, h

def fit_to_jpeg_limits(w: int, h: int) -> Tuple[int, int, float]:
    scale = 1.0
    if w > JPEG_MAX_SIDE or h > JPEG_MAX_SIDE:
        scale = min(JPEG_MAX_SIDE / float(w), JPEG_MAX_SIDE / float(h))
        w = max(1, int(math.floor(w * scale)))
        h = max(1, int(math.floor(h * scale)))
    return w, h, scale

def area_sequence_linear(min_area: int, max_area: int, count: int):
    if count <= 0:
        return []
    if count == 1:
        return [int(round((min_area + max_area) / 2.0))]
    step = (max_area - min_area) / float(count - 1)
    return [int(round(min_area + i * step)) for i in range(count)]

# ---- Worker (TOP-LEVEL for Windows pickling) ----
def process_one(item):
    """
    item = (fname, w, h, realized_area, target_area, quality, r, g, b)
    Low-memory: create a solid-color PIL image and save directly (no giant raw buffers).
    """
    fname, w, h, realized_area, target_area, quality, r, g, b = item
    img = Image.new("RGB", (w, h), color=(r, g, b))
    # Progressive + optimize for decent speed/size tradeoff
    img.save(fname, format="JPEG", quality=quality, optimize=True, progressive=True)
    return (fname, w, h, realized_area, target_area)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--min_area", type=float, default=1000.0)
    ap.add_argument("--max_area", type=float, default=1e10)
    ap.add_argument("--min_ar", type=float, default=0.5)
    ap.add_argument("--max_ar", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--outdir", type=str, default=".")
    ap.add_argument("--prefix", type=str, default="image")
    ap.add_argument("--start_index", type=int, default=0)
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))  # safer default

    args = ap.parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    os.makedirs(args.outdir, exist_ok=True)

    min_area = max(1, int(math.floor(args.min_area)))
    max_area_user = int(math.floor(args.max_area))
    max_area = clip_max_area_for_jpeg(max_area_user)
    if max_area < max_area_user:
        print(f"[warn] Clipping max_area from {max_area_user} to {max_area} (side limit {JPEG_MAX_SIDE}).", file=sys.stderr)
    if min_area > max_area:
        raise ValueError(f"min_area ({min_area}) > max_area ({max_area})")

    # Linear target areas
    target_areas = area_sequence_linear(min_area, max_area, args.count)

    # Plan dims + realized areas; choose a color per image (deterministic if --seed set)
    planned = []
    for t_area in target_areas:
        ar = choose_aspect_ratio(args.min_ar, args.max_ar)
        w, h = dims_from_area_and_ar(t_area, ar)
        w, h, scale = fit_to_jpeg_limits(w, h)
        realized = w * h
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        planned.append({"target": t_area, "w": w, "h": h, "realized": realized, "color": color})

    # Sort by realized area so image0 is smallest, etc.
    planned.sort(key=lambda x: (x["realized"], x["w"], x["h"]))

    if planned:
        print(f"[info] Smallest planned: {planned[0]['w']}x{planned[0]['h']} area={planned[0]['realized']:,} (target≈{planned[0]['target']:,})")
        print(f"[info] Largest planned : {planned[-1]['w']}x{planned[-1]['h']} area={planned[-1]['realized']:,} (target≈{planned[-1]['target']:,})")

    # Build work items with final filenames assigned by ascending size
    work_items = []
    for idx, p in enumerate(planned, start=args.start_index):
        fname = os.path.join(args.outdir, f"{args.prefix}{idx}.jpg")
        r, g, b = p["color"]
        work_items.append((fname, p["w"], p["h"], p["realized"], p["target"], args.quality, r, g, b))

    # Parallel encode (each worker only holds one image in memory)
    results = []
    workers = max(1, args.workers)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process_one, item) for item in work_items]
        for fut in as_completed(futures):
            results.append(fut.result())

    # Report in filename order
    results.sort(key=lambda r: r[0])
    for fname, w, h, realized_area, target_area in results:
        print(f"Wrote {fname}: {w}x{h}  area={realized_area:,}  target≈{target_area:,}")

if __name__ == "__main__":
    main()
