import argparse
import random
import shutil
from pathlib import Path

FIXED_MIN = 0
FIXED_MAX = 4319  # inclusive
FIXED_COUNT = FIXED_MAX - FIXED_MIN + 1  # 4320

def ensure_sources(src_dir: Path):
    names = ["image0.jpg", "image1.jpg", "image2.jpg", "image3.jpg"]
    srcs, missing = [], []
    for n in names:
        p = src_dir / n
        (srcs if p.is_file() else missing).append(p)
    if missing:
        raise FileNotFoundError("Missing source files:\n  " + "\n  ".join(map(str, missing)))
    return srcs

def pick_indices(total: int, rng: random.Random):
    """Return 'total' indices in [0,4319]. Unique until 4320 slots are used, then with replacement."""
    base = list(range(FIXED_MIN, FIXED_MAX + 1))
    rng.shuffle(base)
    if total <= FIXED_COUNT:
        return base[:total]
    extra = [rng.randint(FIXED_MIN, FIXED_MAX) for _ in range(total - FIXED_COUNT)]
    return base + extra

def main():
    ap = argparse.ArgumentParser(description="Copy image0/1/2/3.jpg to random image{idx}.jpg where idx in [0..4319].")
    ap.add_argument("--src-dir", default=".", help="Directory containing image0.jpg..image3.jpg")
    ap.add_argument("--dst-dir", default="synthImage", help="Destination directory (will be created)")
    ap.add_argument("--per-image", type=int, default=100, help="Copies to make per source image (total = 4×per-image)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    ap.add_argument("--choose-src", choices=["roundrobin", "random"], default="roundrobin",
                    help="How to choose which source image to copy each time")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without copying")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    sources = ensure_sources(src_dir)
    total = len(sources) * args.per_image  # 4 * per-image
    indices = pick_indices(total, rng)

    print(f"[plan] sources=4 per-image={args.per_image} total={total} index-range=[{FIXED_MIN},{FIXED_MAX}]")
    if args.dry_run:
        print("[dry-run] no files will be written")

    written = 0
    for i in range(total):
        src = sources[i % len(sources)] if args.choose_src == "roundrobin" else rng.choice(sources)
        idx = indices[i]
        dst = dst_dir / f"image{idx}.jpg"
        if args.dry_run:
            print(f"[dry-run] {src} -> {dst}")
        else:
            shutil.copy2(src, dst)
        written += 1

    print(f"[done] wrote {written} files into {dst_dir}")

if __name__ == "__main__":
    main()