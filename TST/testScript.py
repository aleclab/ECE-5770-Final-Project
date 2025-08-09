import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# WB timer parsing
# -----------------------------------------------------------------------------
# We match timers by substring on the "message" field (case-insensitive).
# Make sure these substrings appear in your wbTime_stop(Compute, "...") labels.
WB_TIMER_SUBSTR = {
    "sharpen":  ["sharpen"],                  # e.g., "Sharpen kernel"
    "sobel":    ["sobel"],                    # e.g., "Sobel fused kernel"
    "gaussian": ["gaussian horiz", "gaussian vert", "gaussian kernel", "gaussian"],
}

RE_CORRECT = re.compile(r'"correctq"\s*:\s*true', re.IGNORECASE)

def parse_wb_timers_all(text: str):
    """
    Return list of (message_lower, ms) for every WB 'timer' JSON line found.
    WB puts elapsed_time in nanoseconds -> convert to milliseconds.
    """
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s or not s.startswith("{"):
            continue
        try:
            rec = json.loads(s)
        except json.JSONDecodeError:
            continue
        if rec.get("type") != "timer":
            continue
        data = rec.get("data", {})
        msg = str(data.get("message", "")).strip().lower()
        ns = data.get("elapsed_time", None)
        if isinstance(ns, (int, float)):
            out.append((msg, float(ns) / 1e6))
    return out

def parse_time_ms(op: str, text: str, debug: bool = False) -> float | None:
    """
    Sum WB timers whose 'message' contains any configured substring for this op.
    Returns total ms or None if nothing matched.
    """
    timers = parse_wb_timers_all(text)
    needles = [s.lower() for s in WB_TIMER_SUBSTR.get(op, [])]
    total = 0.0
    matched = False
    for msg, ms in timers:
        if any(n in msg for n in needles):
            total += ms
            matched = True
    if debug:
        uniq = sorted(set(m for m, _ in timers))
        print(f"[debug] WB timers seen ({len(uniq)}):")
        for m in uniq:
            print("  -", m)
        if not matched:
            print(f"[debug] no match for op='{op}' using needles={needles}")
    return total if matched else None

# -----------------------------------------------------------------------------
# OpenCV expected images (used to write expected_*.ppm into dataset)
# -----------------------------------------------------------------------------
def expected_gaussian(img_bgr):
    # 5x5, sigmaX=0, BORDER_REFLECT_101 (OpenCV default)
    return cv2.GaussianBlur(img_bgr, (5, 5), 0)

def expected_sobel(img_bgr):
    # BGR->GRAY, Sobel 3x3 CV_64F, L2 magnitude, clip to [0,255], then make 3-channel
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(sx, sy)
    mag8 = np.uint8(np.clip(mag, 0, 255))
    return cv2.cvtColor(mag8, cv2.COLOR_GRAY2BGR)

def expected_sharpen(img_bgr):
    # 3x3 sharpen with center=9, neighbors=-1, BORDER_REFLECT_101 default
    k = np.array([[-1, -1, -1],
                  [-1,  9, -1],
                  [-1, -1, -1]], dtype=np.float32)
    return cv2.filter2D(img_bgr, ddepth=-1, kernel=k, borderType=cv2.BORDER_DEFAULT)

# -----------------------------------------------------------------------------
# Filesystem helpers
# -----------------------------------------------------------------------------
def list_images(images_dir: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".ppm"}
    files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files

def write_ppm(path: Path, img_bgr):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img_bgr)
    if not ok:
        raise RuntimeError(f"Failed to write {path}")

# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------
def run_one(exe: Path,
            input_ppm: Path,
            expected_ppm: Path,
            op: str,
            trailing_comma_in_i: bool,
            extra_i: list[str] | None,
            print_cmd: bool = False,
            debug: bool = False):
    """
    Launch the exe once and return (correct: bool, time_ms: float|None, stdout: str)
    """
    inputs = [str(input_ppm)]
    if extra_i:
        inputs.extend(extra_i)
    i_arg = ",".join(inputs) + ("," if trailing_comma_in_i else "")

    cmd = [str(exe), "--op", op, "-i", i_arg, "-e", str(expected_ppm), "-t", "image"]
    if print_cmd:
        print("[cmd]", " ".join(cmd))

    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    correct = RE_CORRECT.search(out) is not None
    t_ms = parse_time_ms(op, out, debug=debug)
    return correct, t_ms, out

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_times(op: str, times_ms, out_dir: Path):
    if not times_ms:
        return
    xs = list(range(len(times_ms)))
    avg = sum(times_ms) / len(times_ms)
    plt.figure()
    plt.bar(xs, times_ms)
    plt.axhline(avg)
    plt.title(f"{op} times per image (ms), avg={avg:.3f} ms")
    plt.xlabel("image index")
    plt.ylabel("time (ms)")
    plt.tight_layout()
    out_path = out_dir / f"{op}_times.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[plot] wrote {out_path}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Prepare dataset, run CUDA exe for multiple filters, and plot timings.")
    ap.add_argument("--exe", required=True, help="Path to your built .exe")
    ap.add_argument("--images", required=True, help="Directory containing source images")
    ap.add_argument("--dataset", required=True, help="Directory to write dataset/<idx> subfolders")
    ap.add_argument("--ops", nargs="+", default=["gaussian", "sobel", "sharpen"],
                    help="Filters to run (subset of: gaussian sobel sharpen)")
    ap.add_argument("--start-idx", type=int, default=0, help="Starting dataset index")
    ap.add_argument("--max", type=int, default=None, help="Max number of images to process (after sorting)")
    ap.add_argument("--trailing-comma-in-i", action="store_true",
                    help="Append trailing comma to -i CSV (some WB harnesses expect this).")
    ap.add_argument("--extra-i", nargs="*", default=None,
                    help="Extra inputs to append to -i after the PPM (rare; pass absolute paths).")
    ap.add_argument("--out-plots", default=".", help="Directory to save timing plots")
    ap.add_argument("--print-cmd", action="store_true", help="Print each exe command before running")
    ap.add_argument("--debug", action="store_true", help="Print parsed WB timer messages for troubleshooting")
    args = ap.parse_args()

    exe = Path(args.exe)
    images_dir = Path(args.images)
    dataset_dir = Path(args.dataset)
    plots_dir = Path(args.out_plots)

    # Early sanity checks
    if not exe.is_file():
        print(f"[err] --exe not found: {exe}", file=sys.stderr)
        sys.exit(2)
    if not images_dir.exists():
        print(f"[err] --images not found: {images_dir}", file=sys.stderr)
        sys.exit(2)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1) Gather and (optionally) limit images
    files = list_images(images_dir)
    if args.max is not None:
        files = files[:args.max]
    if not files:
        print(f"[err] no images found in {images_dir}", file=sys.stderr)
        sys.exit(2)

    # 2) Build dataset/<idx>/ with input<idx>.ppm and expected_<op>.ppm
    print("[prep] building dataset folders and expected outputs...")
    for idx, src in enumerate(files, start=args.start_idx):
        img = cv2.imread(str(src))  # BGR uint8
        if img is None:
            print(f"[warn] failed to read {src}, skipping")
            continue
        d = dataset_dir / f"{idx}"
        d.mkdir(parents=True, exist_ok=True)

        # input PPM
        input_ppm = d / f"input{idx}.ppm"
        write_ppm(input_ppm, img)

        # expected per op
        if "gaussian" in args.ops:
            write_ppm(d / "expected_gaussian.ppm", expected_gaussian(img))
        if "sobel" in args.ops:
            write_ppm(d / "expected_sobel.ppm", expected_sobel(img))
        if "sharpen" in args.ops:
            write_ppm(d / "expected_sharpen.ppm", expected_sharpen(img))

    # 3) Run exe for each idx/op, collect correctness and times
    print("[run] launching exe over all indices and ops...")
    times = {op: [] for op in args.ops}
    n_total = 0
    n_correct = 0

    for idx, src in enumerate(files, start=args.start_idx):
        d = dataset_dir / f"{idx}"
        input_ppm = d / f"input{idx}.ppm"
        if not input_ppm.exists():
            print(f"[warn] missing {input_ppm}, skipping index {idx}")
            continue

        # Prepare per-index extra inputs if any (no special formatting here; pass absolute paths if needed)
        extra_i = list(args.extra_i) if args.extra_i else None

        for op in args.ops:
            expected_ppm = d / f"expected_{op}.ppm"
            if not expected_ppm.exists():
                print(f"[warn] expected not found for op={op}: {expected_ppm} (still running)")
            try:
                ok, t_ms, out_text = run_one(
                    exe=exe,
                    input_ppm=input_ppm,
                    expected_ppm=expected_ppm,
                    op=op,
                    trailing_comma_in_i=args.trailing_comma_in_i,
                    extra_i=extra_i,
                    print_cmd=args.print_cmd,
                    debug=args.debug,
                )
            except subprocess.CalledProcessError as e:
                print(f"[err] run failed for idx={idx} op={op}\n{e.output}")
                continue

            n_total += 1
            if ok:
                n_correct += 1
            else:
                print(f"[fail] idx={idx} op={op} incorrect")

            if t_ms is None:
                print(f"[warn] no timing parsed for idx={idx} op={op}")
            else:
                times[op].append(t_ms)

    print(f"\nSummary: {n_correct} / {n_total} correct")

    # 4) Plot per-op timings and print averages
    for op in args.ops:
        op_times = times[op]
        if not op_times:
            print(f"[info] no times for op={op}")
            continue
        avg = sum(op_times) / len(op_times)
        print(f"[avg] {op}: {avg:.3f} ms over {len(op_times)} runs")
        plot_times(op, op_times, plots_dir)

if __name__ == "__main__":
    main()