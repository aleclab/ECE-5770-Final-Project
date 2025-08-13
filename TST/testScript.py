import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from collections import defaultdict, OrderedDict
import statistics
import cv2
import numpy as np
import os 
import math

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

CUDA_VARIANT_LABEL = {
    0: "CUDA v0 original",
    1: "CUDA v1 ro+const",
    2: "CUDA v2 tiled RGB",
    3: "CUDA v3 tiled Y"
}

COPY_NEEDLES = ["h2d", "d2h"]

SCALE_NEEDLES = ["scale input 0..1 -> 0..255", "scale output 0..255 -> 0..1"]

OP_MARKER = {
    "sobel": "o",
    "sharpen": "s",
    "gaussian": "^",
}
BACKEND_COLOR = {
    "cuda":     "C0",
    "cpu":      "C1",
    "mt":       "C2",
    # distinct colors for Sobel CUDA variants
    "cuda_v0":  "C3",
    "cuda_v1":  "C4",
    "cuda_v2":  "C5",
    "cuda_v3":  "C6",
}

# ----------------------------- WB timer parsing ------------------------------
WB_TIMER_SUBSTR = {
    "sharpen":  ["sharpen kernel"],
    "sobel":    ["sobel fused kernel"],
    "gaussian": ["gaussian horiz", "gaussian vert", "gaussian kernel"],
}
RE_CORRECT = re.compile(r'"correctq"\s*:\s*true', re.IGNORECASE)
# Look for our print like: "CPU backend: MT, threads=12"
RE_THREADS = re.compile(r"threads\s*=\s*(\d+)", re.IGNORECASE)

def parse_wb_timers_all(text: str):
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
            out.append((msg, float(ns) / 1e6))  # ns -> ms
    return out

def parse_time_ms(op: str, text: str, debug: bool = False) -> float | None:
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

def parse_threads_used(text: str) -> int | None:
    m = RE_THREADS.search(text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None

# ------------------------------ expected images ------------------------------
def expected_gaussian(img_bgr):
    return cv2.GaussianBlur(img_bgr, (5, 5), 0)

def expected_sobel(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(sx, sy)
    mag8 = np.uint8(np.clip(mag, 0, 255))
    return cv2.cvtColor(mag8, cv2.COLOR_GRAY2BGR)

def expected_sharpen(img_bgr):
    k = np.array([[-1, -1, -1],
                  [-1,  9, -1],
                  [-1, -1, -1]], dtype=np.float32)
    return cv2.filter2D(img_bgr, ddepth=-1, kernel=k, borderType=cv2.BORDER_DEFAULT)

# --------------------------------- fs helpers --------------------------------
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

# ----------------------------------- run exe ---------------------------------
def run_one(exe: Path,
            input_ppm: Path,
            expected_ppm: Path,
            op: str,
            backend: str,
            threads: int | None,
            trailing_comma_in_i: bool,
            extra_i: list[str] | None,
            print_cmd: bool,
            debug: bool,
            win_high_prio: bool,
            cuda_inc_copies: bool,
            cuda_inc_scales: bool,
            cuda_variant: int | None = None):
    inputs = [str(input_ppm)]
    if extra_i:
        inputs.extend(extra_i)
    i_arg = ",".join(inputs) + ("," if trailing_comma_in_i else "")

    cmd = [str(exe), "--op", op, "--backend", backend, "-i", i_arg, "-e", str(expected_ppm), "-t", "image"]
    if backend == "mt" and threads is not None:
        cmd += ["--threads", str(threads)]
    # NEW: pass sobel variant if requested
    if backend == "cuda" and op == "sobel" and cuda_variant is not None:
        cmd += ["--cuda-variant", str(cuda_variant)]
    if print_cmd:
        print("[cmd]", " ".join(cmd))

    # High-priority on Windows (optional)
    kwargs = {"text": True, "stderr": subprocess.STDOUT}
    try:
        import os
        if win_high_prio and os.name == "nt":
            kwargs["creationflags"] = subprocess.HIGH_PRIORITY_CLASS
    except Exception:
        pass

    out = subprocess.check_output(cmd, **kwargs)
    correct = RE_CORRECT.search(out) is not None
    # Use combined timing (CUDA can include copies/scales per flags)
    t_ms = compute_time_ms(op, backend, out,
                           include_copies=cuda_inc_copies,
                           include_scales=cuda_inc_scales,
                           debug=debug)
    threads_used = parse_threads_used(out)
    return correct, t_ms, threads_used, out

# ---------------------------------- plotting ---------------------------------
BACKEND_DISPLAY = OrderedDict([
    ("cuda", "CUDA"),
    ("cpu",  "CPU"),
    ("mt",   "CPU MT"),
])

def plot_grouped_per_image(op: str, indices, times_by_backend, out_dir: Path):
    if not indices:
        return
    backends = [b for b in BACKEND_DISPLAY.keys() if b in times_by_backend]
    if not backends:
        return

    x = np.arange(len(indices))
    width = 0.8 / max(1, len(backends))  # total 80% width
    plt.figure()
    for i, bk in enumerate(backends):
        series = times_by_backend[bk]
        if not series:
            continue
        pos = x - 0.4 + width/2 + i*width
        plt.bar(pos, series, width, label=BACKEND_DISPLAY[bk])
    plt.xticks(x, [str(i) for i in indices], rotation=0)
    plt.title(f"{op}: per-image kernel time (ms)")
    plt.xlabel("image index")
    plt.ylabel("time (ms)")
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / f"{op}_per_image_grouped.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[plot] wrote {out_path}")


def compute_backend_stats(op: str, results, backends):
    # Returns dict: backend -> {"mean": float, "stdev": float, "n": int}
    stats = {}
    indices = sorted(results[op].keys())
    for bk in backends:
        vals = [results[op][idx].get(bk, {}).get("ms", None) for idx in indices]
        vals = [v for v in vals if v is not None]
        if vals:
            m = sum(vals) / len(vals)
            s = statistics.stdev(vals) if len(vals) > 1 else 0.0
            stats[bk] = {"mean": m, "stdev": s, "n": len(vals)}
    return stats

def plot_avg_per_backend(op: str, stats_by_backend, out_dir: Path):
    if not stats_by_backend:
        print(f"[plot] no data to plot for {op}")
        return

    # Order: CPU, MT, CUDA, then any cuda_v* in numeric order
    order_keys = []
    for k in ["cpu", "mt", "cuda"]:
        if k in stats_by_backend:
            order_keys.append(k)
    variant_keys = [k for k in stats_by_backend if k.startswith("cuda_v")]
    variant_keys.sort(key=lambda s: int(s.split("cuda_v", 1)[1]))
    order_keys.extend([k for k in variant_keys if k in stats_by_backend])

    if not order_keys:
        print(f"[plot] no data to plot for {op}")
        return

    labels  = [BACKEND_DISPLAY.get(k, k) for k in order_keys]
    means   = [stats_by_backend[k]["mean"]  for k in order_keys]
    stdevs  = [stats_by_backend[k]["stdev"] for k in order_keys]
    ns      = [stats_by_backend[k]["n"]     for k in order_keys]
    colors  = [BACKEND_COLOR.get(k, "C7")   for k in order_keys]

    n = len(labels)
    # Grow BOTH width and height with content
    max_label_len = max((len(s) for s in labels), default=10)
    width_in  = max(10.0, 0.8 * n + 0.35 * max_label_len)   # wider with more bars / longer labels
    height_in = max(6.0, 0.6 * n + 2.0)                     # taller with more bars
    dpi = 160                                               # higher DPI so PNG really grows in pixels

    # Build figure/axes explicitly and FORCE size
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
    fig.set_size_inches(width_in, height_in, forward=True)

    y = np.arange(n)
    ax.barh(y, means, xerr=stdevs, capsize=4, color=colors)

    # Make room on the right for text annotations
    xmax = max(means) if means else 1.0
    ax.set_xlim(0, xmax * 1.25)

    # Longest label space on the left; top = best
    ax.invert_yaxis()
    ax.set_yticks(y, labels)
    ax.set_xlabel("kernel time (ms)")
    ax.set_title(f"{op}: average kernel time across images (N shown)")

    # Annotate to the right of bars
    offset = max(0.01 * xmax, 0.5)
    for yi, (m, n_i) in enumerate(zip(means, ns)):
        ax.text(m + offset, yi, f"{m:.2f} ms  (N={n_i})", va="center", fontsize=9)

    # Generous margins; tight bbox at save time
    fig.tight_layout()
    plt.subplots_adjust(left=0.25, right=0.98, top=0.9, bottom=0.12)

    out_path = out_dir / f"{op}_avg_by_backend.png"
    # Force dpi and 'tight' so you get the requested pixel size and no clipping
    fig.savefig(out_path, dpi=dpi,  pad_inches=0.2)
    plt.close(fig)
    print(f"[plot] wrote {out_path}  (figsize={width_in:.1f}x{height_in:.1f} in @ {dpi} dpi)")

def plot_scatter_time_vs_area(rows, out_path: Path):
    # Group points by (backend, op), then scatter once per group (fast).
    # cap total points to keep plotting relatively quick on huge datasets.
    xs, ys, cs, ms = [], [], [], []
    color_keys_present = set()
    ops_present = set()

    for r in rows:
        t_str = r.get("time_ms", "")
        if not t_str:
            continue
        try:
            t = float(t_str)
        except ValueError:
            continue
        area = r.get("area", None)
        if area is None:
            continue

        op = r.get("op", "")
        bk = r.get("backend", "")
        cv = r.get("cuda_variant", "")

        # Pick color key: split CUDA Sobel by variant
        if bk == "cuda" and op == "sobel" and str(cv) != "":
            color_key = f"cuda_v{cv}"
        else:
            color_key = bk

        xs.append(area)
        ys.append(t)
        cs.append(BACKEND_COLOR.get(color_key, "C7"))  # fallback color
        ms.append(OP_MARKER.get(op, "x"))
        color_keys_present.add(color_key)
        ops_present.add(op)

    if not xs:
        print("[scatter] no datapoints to plot")
        return

    plt.figure(figsize=(9, 6))
    # scatter points
    for x, y, c, m in zip(xs, ys, cs, ms):
        plt.scatter(x, y, c=c, marker=m, alpha=0.5, s=16, linewidths=0)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("image area (width × height, pixels)")
    plt.ylabel("kernel time (ms, log scale)")
    plt.title("Kernel time vs image area (color=backend/variant, marker=op)")

    # Legends: colors for (backend/variant), markers for op
    color_handles = []
    # Use BACKEND_DISPLAY for nice labels if available
    for key in sorted(color_keys_present):
        label = BACKEND_DISPLAY.get(key, key)
        color_handles.append(
            plt.Line2D([0],[0], color=BACKEND_COLOR.get(key,"C7"),
                       marker='o', linestyle='None', label=label)
        )

    marker_handles = []
    for op in sorted(ops_present):
        marker_handles.append(
            plt.Line2D([0],[0], color="black",
                       marker=OP_MARKER.get(op, "x"),
                       linestyle='None', label=op)
        )

    leg1 = plt.legend(handles=color_handles, title="Backend / Variant", loc="upper left")
    plt.gca().add_artist(leg1)
    plt.legend(handles=marker_handles, title="Op", loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[plot] wrote {out_path}")
    
def backend_sort_key(bk):
    if bk == "cpu": return (0, 0)
    if bk == "mt":  return (1, 0)
    if bk.startswith("cuda_v"):
        try:
            v = int(bk.split("cuda_v",1)[1])
        except:
            v = 99
        return (2, v)
    if bk == "cuda": return (3, 0)
    return (9, 0)

def finalize_outputs(results, rows_for_csv, args, dims, plots_dir: Path):
    print("\n[finalize] generating plots/CSV from collected results...")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Partial summary based on collected rows
    n_total = len(rows_for_csv)
    n_correct = sum(int(r.get("correct", 0)) for r in rows_for_csv)
    print(f"Summary (partial ok): {n_correct} / {n_total} runs reported correctq:true")

    # Average plots per filter (horizontal bars, dynamic size)
    for op in args.ops:
        indices = sorted(results[op].keys())
        backend_keys = []
        for idx in indices:
            backend_keys.extend(results[op][idx].keys())
        backend_keys = sorted(set(backend_keys))
        backend_order = sorted(backend_keys, key=backend_sort_key)

        labels, means, stdevs, ns, colors = [], [], [], [], []
        for bk in backend_order:
            vals = [results[op][idx].get(bk, {}).get("ms", None) for idx in indices]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            labels.append(BACKEND_DISPLAY.get(bk, bk))
            means.append(float(np.mean(vals)))
            stdevs.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
            ns.append(len(vals))
            # color per backend/variant (falls back to C7 if missing)
            colors.append(BACKEND_COLOR.get(bk, "C7"))

        if not labels:
            print(f"[plot] no data to plot for {op}")
            continue

        # ---- horizontal bar chart with dynamic figure size ----
        n = len(labels)
        max_label_len = max((len(s) for s in labels), default=10)
        width_in  = max(12.0, 0.9 * n + 0.45 * max_label_len)
        height_in = max(6.0,  0.6 * n + 2.0)
        dpi = 160

        fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
        fig.set_size_inches(width_in, height_in, forward=True)
        y = np.arange(n)
        ax.barh(y, means, xerr=stdevs, capsize=4, color=colors)

        # axis/labels
        ax.invert_yaxis()  # top entry at top
        ax.set_yticks(y, labels)
        ax.set_xlabel("kernel time (ms)")
        ax.set_title(f"{op}: average kernel time across images (N shown)")

        # room for annotations on the right
        xmax = max(means) if means else 1.0
        ax.set_xlim(0, xmax * 1.30)
        offset = max(0.01 * xmax, 0.5)
        for yi, (m, n_runs) in enumerate(zip(means, ns)):
            ax.text(m + offset, yi, f"{m:.2f} ms  (N={n_runs})", va="center", fontsize=9)

        # generous margins; then save
        plt.subplots_adjust(left=0.30, right=0.98, top=0.90, bottom=0.12)
        fig.tight_layout()
        out_path = plots_dir / f"{op}_avg_by_backend.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)
        print(f"[plot] wrote {out_path}  (figsize={width_in:.1f}x{height_in:.1f} in @ {dpi} dpi)")

    # CSV
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "index","image","op","backend","cuda_variant","threads_used","time_ms","correct","width","height","area"
            ])
            writer.writeheader()
            for row in rows_for_csv:
                writer.writerow(row)
        print(f"[csv] wrote {csv_path}")

    # Scatter (optional)
    if args.scatter:
        scatter_path = Path(args.scatter_out) if args.scatter_out else (plots_dir / "time_vs_area.png")
        if scatter_path.is_dir():  # if a directory was passed by mistake
            scatter_path = scatter_path / "time_vs_area.png"
        plot_scatter_time_vs_area(rows_for_csv, scatter_path)

    # Thread count summary (MT)
    for op in args.ops:
        mt_counts = [int(r["threads_used"]) for r in rows_for_csv
                     if r.get("backend") == "mt" and r.get("op") == op and str(r.get("threads_used","")).isdigit()]
        if mt_counts:
            avg = sum(mt_counts) / len(mt_counts)
            uniq = sorted(set(mt_counts))
            print(f"[threads] {op}: CPU MT avg={avg:.2f}, unique={uniq}")
        else:
            print(f"[threads] {op}: no MT runs (or no thread info parsed)")

def compute_time_ms(op: str, backend: str, text: str,
                    include_copies: bool, include_scales: bool,
                    debug: bool = False) -> float | None:
    timers = parse_wb_timers_all(text)  # list of (msg_lower, ms)
    needles = [s.lower() for s in WB_TIMER_SUBSTR.get(op, [])]

    total = 0.0
    matched_kernel = False

    # Sum the kernel timers (e.g., sobel fused, gaussian horiz/vert, sharpen)
    for msg, ms in timers:
        if any(n in msg for n in needles):
            total += ms
            matched_kernel = True

    # Add copies for CUDA (H2D + D2H)
    if backend == "cuda" and include_copies:
        for msg, ms in timers:
            if any(k in msg for k in COPY_NEEDLES):
                total += ms

    # Optionally add CUDA scale kernels (to mirror CPU pre/post scales)
    if backend == "cuda" and include_scales:
        for msg, ms in timers:
            if any(s in msg for s in SCALE_NEEDLES):
                total += ms

    if debug:
        uniq = sorted(set(m for m, _ in timers))
        print(f"[debug] WB timers seen ({len(uniq)}):")
        for m in uniq:
            print("  -", m)
        if not matched_kernel:
            print(f"[debug] no kernel match for op='{op}' using needles={needles}")
        else:
            added = []
            if backend == "cuda" and include_copies: added.append("H2D/D2H")
            if backend == "cuda" and include_scales: added.append("scale in/out")
            if added:
                print(f"[debug] added {' & '.join(added)} to CUDA time")
    return total if matched_kernel else None

def discover_dataset_inputs(dataset_dir: Path):
    """Return (sorted_indices, dict index->input_ppm_path)."""
    indices = []
    inputs = {}
    if not dataset_dir.exists():
        return indices, inputs
    for sub in dataset_dir.iterdir():
        if not sub.is_dir():
            continue
        try:
            idx = int(sub.name)
        except ValueError:
            continue
        # Prefer "input{idx}.ppm"; otherwise any "input*.ppm"
        cand = sub / f"input{idx}.ppm"
        if cand.exists():
            indices.append(idx)
            inputs[idx] = cand
            continue
        alts = list(sub.glob("input*.ppm"))
        if alts:
            indices.append(idx)
            inputs[idx] = alts[0]
    indices.sort()
    return indices, inputs

# ------------------------------------ main ---------- ------------------------
def main():
    ap = argparse.ArgumentParser(description="Prepare dataset, run CUDA/CPU/MT backends, parse WB timers, plot grouped comparisons, export CSV.")
    ap.add_argument("--exe", required=True, help="Path to built .exe")
    ap.add_argument("--skip-prep", action="store_true",
                    help="Skip preprocessing: do not generate PPMs/expected images; reuse existing Dataset/<idx>/ subfolders.")
    ap.add_argument("--images", required=True, help="Directory containing source images")
    ap.add_argument("--dataset", required=True, help="Directory to write dataset/<idx> subfolders")
    ap.add_argument("--ops", nargs="+", default=["gaussian", "sobel", "sharpen"],
                    help="Filters to run (subset of: gaussian sobel sharpen)")
    ap.add_argument("--backends", nargs="+", default=["cuda", "cpu", "mt"],
                    help="Backends to run (subset of: cuda cpu mt)")
    ap.add_argument("--threads", type=int, default=None,
                    help="Thread count for backend=mt (if omitted, app decides)")
    ap.add_argument("--start-idx", type=int, default=0, help="Starting dataset index")
    ap.add_argument("--max", type=int, default=None, help="Max number of images to process")
    ap.add_argument("--trailing-comma-in-i", action="store_true",
                    help="Append trailing comma to -i CSV (some WB harnesses expect this).")
    ap.add_argument("--extra-i", nargs="*", default=None,
                    help="Extra inputs to append to -i after the PPM (rare; pass absolute paths).")
    ap.add_argument("--out-plots", default=".", help="Directory to save timing plots")
    ap.add_argument("--csv", default=None, help="Optional path to write a CSV of all runs")
    ap.add_argument("--print-cmd", action="store_true", help="Print each exe command before running")
    ap.add_argument("--debug", action="store_true", help="Print parsed WB timer messages for troubleshooting")
    ap.add_argument("--scatter", action="store_true",
                    help="Emit a single scatter plot: time (ms, log) vs image area, colored by backend, shaped by op.")
    ap.add_argument("--scatter-out", default=None,
                    help="Output PNG for the scatter plot (default: <plots_dir>/time_vs_area.png)")
    ap.add_argument("--win-high-priority", action="store_true",
                    help="On Windows, launch the exe with HIGH_PRIORITY_CLASS")
    ap.add_argument("--sobel-variants", nargs="+", default=["2"],
                    help="CUDA Sobel variants to run: 0 1 2 3, or 'all'")

    # CUDA timing options
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--cuda-kernel-only", dest="cuda_inc_copies", action="store_false",
                       help="CUDA timing excludes H2D/D2H (kernel-only)")
    group.add_argument("--cuda-include-copies", dest="cuda_inc_copies", action="store_true",
                       help="CUDA timing includes H2D/D2H")
    ap.set_defaults(cuda_inc_copies=True)
    ap.add_argument("--cuda-include-scales", action="store_true",
                    help="Also include CUDA pre/post scale kernels in CUDA time")

    args = ap.parse_args()

    # Normalize sobel variants
    if any(s.lower() == "all" for s in args.sobel_variants):
        sobel_variants = [0, 1, 2, 3]
    else:
        sobel_variants = []
        for s in args.sobel_variants:
            try:
                v = int(s)
                if v in (0, 1, 2, 3):
                    sobel_variants.append(v)
            except ValueError:
                pass
        if not sobel_variants:
            sobel_variants = [2]  # default

    # Extend BACKEND_DISPLAY with nice labels for the requested variants
    for v in sobel_variants:
        BACKEND_DISPLAY[f"cuda_v{v}"] = CUDA_VARIANT_LABEL.get(v, f"CUDA v{v}")

    # Safe ops list + echo
    ops_to_run = [op for op in args.ops if op in ("sobel", "sharpen", "gaussian")]
    if not ops_to_run:
        print("[err] no valid ops requested (expect any of: sobel sharpen gaussian)", file=sys.stderr)
        sys.exit(2)
    print(f"[args] ops={ops_to_run} backends={args.backends} sobel_variants={sobel_variants}")

    exe = Path(args.exe)
    images_dir = Path(args.images)
    dataset_dir = Path(args.dataset)
    plots_dir = Path(args.out_plots)

    if not exe.is_file():
        print(f"[err] --exe not found: {exe}", file=sys.stderr); sys.exit(2)
    if not args.skip_prep and not images_dir.exists():
        print(f"[err] --images not found: {images_dir}", file=sys.stderr); sys.exit(2)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Results & CSV rows (must exist for both prep paths)
    results = defaultdict(lambda: defaultdict(dict))
    rows_for_csv = []

    # Prep dataset & capture dimensions
    dims = {}  # idx -> (w,h)
    run_items = []  # list of tuples (idx, input_ppm_path, image_name_for_csv)

    if args.skip_prep:
        print("[prep] skipping preprocessing; discovering existing Dataset/<idx>/input*.ppm ...")
        idxs, input_map = discover_dataset_inputs(dataset_dir)
        if args.max is not None:
            idxs = idxs[:args.max]
        if not idxs:
            print(f"[err] --skip-prep set but no inputs found in {dataset_dir}", file=sys.stderr); sys.exit(2)

        for idx in idxs:
            in_ppm = input_map[idx]
            img = cv2.imread(str(in_ppm))
            if img is None:
                print(f"[warn] failed to read {in_ppm}, skipping idx={idx}")
                continue
            h, w = img.shape[:2]
            dims[idx] = (w, h)
            run_items.append((idx, in_ppm, in_ppm.name))

    else:
        # Build from source images
        files = list_images(images_dir)
        if args.max is not None:
            files = files[:args.max]
        if not files:
            print(f"[err] no images found in {images_dir}", file=sys.stderr); sys.exit(2)

        print("[prep] building dataset folders and expected outputs...")
        for idx, src in enumerate(files, start=args.start_idx):
            img = cv2.imread(str(src))
            if img is None:
                print(f"[warn] failed to read {src}, skipping"); continue
            h, w = img.shape[:2]
            dims[idx] = (w, h)
            d = dataset_dir / f"{idx}"
            d.mkdir(parents=True, exist_ok=True)

            # Write input ppm
            input_ppm = d / f"input{idx}.ppm"
            write_ppm(input_ppm, img)

            # Write expected outputs only for requested ops
            if "gaussian" in args.ops:
                write_ppm(d / "expected_gaussian.ppm", expected_gaussian(img))
            if "sobel" in args.ops:
                write_ppm(d / "expected_sobel.ppm", expected_sobel(img))
            if "sharpen" in args.ops:
                write_ppm(d / "expected_sharpen.ppm", expected_sharpen(img))

            run_items.append((idx, input_ppm, Path(src).name))

    # Run all combinations
    print("[run] launching exe over all indices, backends, and ops...")
    n_total = 0
    n_correct = 0

    try:
        for (idx, input_ppm, image_name) in run_items:
            d = dataset_dir / f"{idx}"

            for op in ops_to_run:
                expected_ppm = d / f"expected_{op}.ppm"
                if not expected_ppm.exists():
                    print(f"[warn] expected not found for op={op}: {expected_ppm} (still running)")

                # run all requested backends for this (idx, op)
                for backend in args.backends:
                    if op == "sobel" and backend == "cuda":
                        # Fan out CUDA Sobel variants
                        for v in sobel_variants:
                            try:
                                ok, t_ms, threads_used, out_text = run_one(
                                    exe=exe,
                                    input_ppm=input_ppm,
                                    expected_ppm=expected_ppm,
                                    op=op,
                                    backend="cuda",
                                    threads=args.threads,
                                    trailing_comma_in_i=args.trailing_comma_in_i,
                                    extra_i=args.extra_i,
                                    print_cmd=args.print_cmd,
                                    debug=args.debug,
                                    win_high_prio=args.win_high_priority,
                                    cuda_inc_copies=args.cuda_inc_copies,
                                    cuda_inc_scales=args.cuda_include_scales,
                                    cuda_variant=v,
                                )
                            except subprocess.CalledProcessError as e:
                                print(f"[err] run failed for idx={idx} op={op} backend=cuda_v{v}\n{e.output}")
                                results[op][idx][f"cuda_v{v}"] = {"ok": False, "ms": None, "threads": None}
                                continue

                            n_total += 1
                            if ok: n_correct += 1
                            print(f"[res] idx={idx} op={op:8s} backend={'cuda_v'+str(v):8s} -> {'OK' if ok else 'MISMATCH'}"
                                  + ("" if t_ms is None else f", time={t_ms:.3f} ms")
                                  + ", threads=0")
                            results[op][idx][f"cuda_v{v}"] = {"ok": ok, "ms": t_ms, "threads": 0}

                            w, h = dims.get(idx, (0, 0))
                            rows_for_csv.append({
                                "index": idx, "image": image_name, "op": op,
                                "backend": "cuda", "cuda_variant": v,
                                "threads_used": 0,
                                "time_ms": f"{t_ms:.6f}" if t_ms is not None else "",
                                "correct": int(bool(ok)), "width": w, "height": h, "area": w * h
                            })
                    else:
                        # Single run for non-CUDA or non-Sobel backends
                        try:
                            ok, t_ms, threads_used, out_text = run_one(
                                exe=exe,
                                input_ppm=input_ppm,
                                expected_ppm=expected_ppm,
                                op=op,
                                backend=backend,
                                threads=args.threads,
                                trailing_comma_in_i=args.trailing_comma_in_i,
                                extra_i=args.extra_i,
                                print_cmd=args.print_cmd,
                                debug=args.debug,
                                win_high_prio=args.win_high_priority,
                                cuda_inc_copies=args.cuda_inc_copies,
                                cuda_inc_scales=args.cuda_include_scales,
                                cuda_variant=None,
                            )
                        except subprocess.CalledProcessError as e:
                            print(f"[err] run failed for idx={idx} op={op} backend={backend}\n{e.output}")
                            results[op][idx][backend] = {"ok": False, "ms": None, "threads": None}
                            continue

                        n_total += 1
                        if ok: n_correct += 1
                        if threads_used is None:
                            if backend == "mt":
                                threads_used = args.threads if args.threads is not None else 0
                            elif backend == "cpu":
                                threads_used = 1
                            else:
                                threads_used = 0
                        print(f"[res] idx={idx} op={op:8s} backend={backend:8s} -> {'OK' if ok else 'MISMATCH'}"
                              + ("" if t_ms is None else f", time={t_ms:.3f} ms")
                              + f", threads={threads_used}")
                        results[op][idx][backend] = {"ok": ok, "ms": t_ms, "threads": threads_used}

                        w, h = dims.get(idx, (0, 0))
                        rows_for_csv.append({
                            "index": idx, "image": image_name, "op": op,
                            "backend": backend, "cuda_variant": "",
                            "threads_used": threads_used,
                            "time_ms": f"{t_ms:.6f}" if t_ms is not None else "",
                            "correct": int(bool(ok)), "width": w, "height": h, "area": w * h
                        })

                # pixel-perfect check after all backends for this (idx, op)
                present_keys = list(results[op][idx].keys())

                # missing by requested backend set (treat any cuda_v* as "cuda")
                missing = []
                for bk in args.backends:
                    if bk == "cuda":
                        if not any(k.startswith("cuda_v") for k in present_keys) and "cuda" not in present_keys:
                            missing.append("cuda")
                    else:
                        if bk not in present_keys:
                            missing.append(bk)

                if not missing:
                    def backend_ok(bk):
                        if bk == "cuda":
                            vks = [k for k in present_keys if k.startswith("cuda_v")]
                            if vks:
                                return any(results[op][idx][k].get("ok", False) for k in vks)
                            return results[op][idx].get("cuda", {}).get("ok", False)
                        return results[op][idx].get(bk, {}).get("ok", False)

                    ok_all = all(backend_ok(bk) for bk in args.backends)
                    if ok_all:
                        print(f"      => ALL BACKENDS MATCH (pixel-perfect vs expected) for idx={idx}, op={op}")
                    else:
                        bad = [bk for bk in args.backends if not backend_ok(bk)]
                        print(f"      => BACKEND(S) FAILED pixel-perfect check: {bad} (idx={idx}, op={op})")
                else:
                    print(f"      => WARNING: missing results for backends {missing} (idx={idx}, op={op})")

    except KeyboardInterrupt:
        print("\n[abort] CTRL+C detected. Finalizing with partial results...")

    finally:
        finalize_outputs(results, rows_for_csv, args, dims, plots_dir)
        
if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise  # let argparse exits propagate (shows usage/errors)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)