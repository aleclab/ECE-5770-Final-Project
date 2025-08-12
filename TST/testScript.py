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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OP_MARKER = {
    "sobel": "o",
    "sharpen": "s",
    "gaussian": "^",
}
BACKEND_COLOR = {
    "cuda": "C0",
    "cpu":  "C1",
    "mt":   "C2",
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
            win_high_prio: bool):
    inputs = [str(input_ppm)]
    if extra_i:
        inputs.extend(extra_i)
    i_arg = ",".join(inputs) + ("," if trailing_comma_in_i else "")

    cmd = [str(exe), "--op", op, "--backend", backend, "-i", i_arg, "-e", str(expected_ppm), "-t", "image"]
    if backend == "mt" and threads is not None:
        cmd += ["--threads", str(threads)]
    if print_cmd:
        print("[cmd]", " ".join(cmd))

    # High-priority on Windows only
    kwargs = {"text": True, "stderr": subprocess.STDOUT}
    try:
        import os
        if win_high_prio and os.name == "nt":
            kwargs["creationflags"] = subprocess.HIGH_PRIORITY_CLASS
    except Exception:
        pass  # fall back silently

    out = subprocess.check_output(cmd, **kwargs)
    correct = RE_CORRECT.search(out) is not None
    t_ms = parse_time_ms(op, out, debug=debug)
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

# def plot_avg_per_backend(op: str, times_by_backend, out_dir: Path):
#     labels, means = [], []
#     for bk in BACKEND_DISPLAY.keys():
#         series = times_by_backend.get(bk, [])
#         if series:
#             labels.append(BACKEND_DISPLAY[bk])
#             means.append(sum(series)/len(series))
#     if not labels:
#         return
#     plt.figure()
#     x = np.arange(len(labels))
#     plt.bar(x, means)
#     plt.xticks(x, labels)
#     plt.title(f"{op}: average kernel time (ms)")
#     plt.ylabel("time (ms)")
#     for xi, m in zip(x, means):
#         plt.text(xi, m, f"{m:.2f}", ha="center", va="bottom")
#     plt.tight_layout()
#     out_path = out_dir / f"{op}_avg_by_backend.png"
#     plt.savefig(out_path)
#     plt.close()
#     print(f"[plot] wrote {out_path}")

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
    labels = []
    means = []
    stdevs = []
    ns = []
    # keep order CUDA, CPU, MT if present
    for bk in ["cuda", "cpu", "mt"]:
        if bk in stats_by_backend:
            labels.append(BACKEND_DISPLAY[bk])
            means.append(stats_by_backend[bk]["mean"])
            stdevs.append(stats_by_backend[bk]["stdev"])
            ns.append(stats_by_backend[bk]["n"])

    x = np.arange(len(labels))
    plt.figure()
    plt.bar(x, means, yerr=stdevs, capsize=4)
    plt.xticks(x, labels)
    plt.ylabel("algorithm execution time (ms)")
    plt.title(f"{op}: average algorithm execution time across images (N shown on bars)")
    for xi, (m, n) in zip(x, zip(means, ns)):
        plt.text(xi, m, f"{m:.2f}\nN={n}", ha="center", va="bottom")
    plt.tight_layout()
    out_path = out_dir / f"{op}_avg_by_backend.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[plot] wrote {out_path}")

def plot_scatter_time_vs_area(rows, out_path: Path):
    # rows are dicts with keys: op, backend, time_ms, area
    xs, ys, cs, ms = [], [], [], []
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
        xs.append(area)
        ys.append(t)
        cs.append(BACKEND_COLOR.get(bk, "C7"))
        ms.append(OP_MARKER.get(op, "x"))

    if not xs:
        print("[scatter] no datapoints to plot")
        return

    plt.figure(figsize=(8, 6))
    # Plot in groups to get a combined legend (backend color + op marker)
    # Build proxy artists for legend
    backend_handles = []
    for bk, label in BACKEND_DISPLAY.items():
        backend_handles.append(plt.Line2D([0],[0], color=BACKEND_COLOR.get(bk,"C7"), marker='o', linestyle='None', label=label))
    op_handles = []
    for op, marker in OP_MARKER.items():
        op_handles.append(plt.Line2D([0],[0], color="black", marker=marker, linestyle='None', label=op))

    # Actually scatter: we can just loop and plot individually (fast enough for a few thousand)
    for x, y, c, m in zip(xs, ys, cs, ms):
        plt.scatter(x, y, c=c, marker=m, alpha=0.5, s=16, linewidths=0)

    plt.xscale("log")  # area can span orders of magnitude too; comment out if you prefer linear X
    plt.yscale("log")
    plt.xlabel("image area (width × height, pixels)")
    plt.ylabel("algorithm execution time (ms, log scale)")
    plt.title("algorithm execution time vs image area (color=backend, marker=op)")
    # Two-part legend
    leg1 = plt.legend(handles=backend_handles, title="Backend", loc="upper left")
    plt.gca().add_artist(leg1)
    plt.legend(handles=op_handles, title="Op", loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[plot] wrote {out_path}")

# ------------------------------------ main -----------------------------------
def main():
    ap = argparse.ArgumentParser(description="Prepare dataset, run CUDA/CPU/MT backends, parse WB timers, plot grouped comparisons, export CSV.")
    ap.add_argument("--exe", required=True, help="Path to your built .exe")
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
    args = ap.parse_args()

    exe = Path(args.exe)
    images_dir = Path(args.images)
    dataset_dir = Path(args.dataset)
    plots_dir = Path(args.out_plots)

    if not exe.is_file():
        print(f"[err] --exe not found: {exe}", file=sys.stderr); sys.exit(2)
    if not images_dir.exists():
        print(f"[err] --images not found: {images_dir}", file=sys.stderr); sys.exit(2)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    files = list_images(images_dir)
    if args.max is not None:
        files = files[:args.max]
    if not files:
        print(f"[err] no images found in {images_dir}", file=sys.stderr); sys.exit(2)

    # Prep dataset & capture dimensions
    dims = {}  # idx -> (w,h)
    print("[prep] building dataset folders and expected outputs...")
    for idx, src in enumerate(files, start=args.start_idx):
        img = cv2.imread(str(src))
        if img is None:
            print(f"[warn] failed to read {src}, skipping"); continue
        h, w = img.shape[:2]
        dims[idx] = (w, h)
        d = dataset_dir / f"{idx}"
        d.mkdir(parents=True, exist_ok=True)
        write_ppm(d / f"input{idx}.ppm", img)
        if "gaussian" in args.ops:
            write_ppm(d / "expected_gaussian.ppm", expected_gaussian(img))
        if "sobel" in args.ops:
            write_ppm(d / "expected_sobel.ppm", expected_sobel(img))
        if "sharpen" in args.ops:
            write_ppm(d / "expected_sharpen.ppm", expected_sharpen(img))

    # results[op][idx][backend] = {"ok": bool, "ms": float or None, "threads": int or None}
    results = defaultdict(lambda: defaultdict(dict))
    rows_for_csv = []

    # Run all combinations
    print("[run] launching exe over all indices, backends, and ops...")
    n_total = 0
    n_correct = 0
    for idx, src in enumerate(files, start=args.start_idx):
        d = dataset_dir / f"{idx}"
        input_ppm = d / f"input{idx}.ppm"
        if not input_ppm.exists():
            print(f"[warn] missing {input_ppm}, skipping index {idx}")
            continue

        for op in args.ops:
            expected_ppm = d / f"expected_{op}.ppm"
            if not expected_ppm.exists():
                print(f"[warn] expected not found for op={op}: {expected_ppm} (still running)")

            for backend in args.backends:
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
                    )
                except subprocess.CalledProcessError as e:
                    print(f"[err] run failed for idx={idx} op={op} backend={backend}\n{e.output}")
                    results[op][idx][backend] = {"ok": False, "ms": None, "threads": None}
                    continue

                n_total += 1
                if ok:
                    n_correct += 1
                    status = "OK"
                else:
                    status = "MISMATCH"

                # Fall back for threads_used if app didn't print it
                if threads_used is None:
                    if backend == "mt":
                        threads_used = args.threads if args.threads is not None else 0
                    elif backend == "cpu":
                        threads_used = 1
                    else:
                        threads_used = 0  # cuda

                print(f"[res] idx={idx} op={op:8s} backend={backend:4s} -> {status}"
                      + ("" if t_ms is None else f", time={t_ms:.3f} ms")
                      + f", threads={threads_used}")

                results[op][idx][backend] = {"ok": ok, "ms": t_ms, "threads": threads_used}

                # Add row for CSV
                w, h = dims.get(idx, (0, 0))
                rows_for_csv.append({
                    "index": idx,
                    "image": Path(src).name,
                    "op": op,
                    "backend": backend,
                    "threads_used": threads_used,
                    "time_ms": f"{t_ms:.6f}" if t_ms is not None else "",
                    "correct": int(bool(ok)),
                    "width": w,
                    "height": h,
                    "area": w * h
                })

            # Pixel-perfect across backends
            present = [bk for bk in args.backends if bk in results[op][idx]]
            all_ok = present and all(results[op][idx][bk]["ok"] for bk in present)
            if len(present) == len(args.backends) and all_ok:
                print(f"      => ALL BACKENDS MATCH (pixel-perfect vs expected) for idx={idx}, op={op}")
            elif len(present) == len(args.backends):
                bad = [bk for bk in args.backends if not results[op][idx][bk]["ok"]]
                print(f"      => BACKEND(S) FAILED pixel-perfect check: {bad} (idx={idx}, op={op})")
            else:
                missing = [bk for bk in args.backends if bk not in present]
                print(f"      => WARNING: missing results for backends {missing} (idx={idx}, op={op})")

    print(f"\nSummary: {n_correct} / {n_total} runs reported correctq:true")

    # Plots (averages only, per filter)
    plots_dir.mkdir(parents=True, exist_ok=True)
    for op in args.ops:
        indices = sorted(results[op].keys())

        # keep a consistent order
        backend_order = [bk for bk in ["cuda", "cpu", "mt"] if bk in args.backends]

        labels, means, stdevs, ns = [], [], [], []
        for bk in backend_order:
            vals = [results[op][idx].get(bk, {}).get("ms", None) for idx in indices]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            labels.append(BACKEND_DISPLAY[bk])
            means.append(float(np.mean(vals)))
            stdevs.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
            ns.append(len(vals))

        if not labels:
            print(f"[plot] no data to plot for {op}")
            continue

        x = np.arange(len(labels))
        plt.figure()
        plt.bar(x, means, yerr=stdevs, capsize=4)
        plt.xticks(x, labels)
        plt.ylabel("kernel time (ms)")
        plt.title(f"{op}: average kernel time across images (N shown)")
        for xi, (m, n) in zip(x, zip(means, ns)):
            plt.text(xi, m, f"{m:.2f}\nN={n}", ha="center", va="bottom")
        plt.tight_layout()
        out_path = plots_dir / f"{op}_avg_by_backend.png"
        plt.savefig(out_path)
        plt.close()
        print(f"[plot] wrote {out_path}")

    # CSV
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "index","image","op","backend","threads_used","time_ms","correct","width","height","area"
            ])
            writer.writeheader()
            for row in rows_for_csv:
                writer.writerow(row)
        print(f"[csv] wrote {csv_path}")

    # Scatter plot (optional)
    if args.scatter:
        scatter_path = Path(args.scatter_out) if args.scatter_out else (plots_dir / "time_vs_area.png")
        plot_scatter_time_vs_area(rows_for_csv, scatter_path)

    # ---- Thread count summary for MT backend ----
    for op in args.ops:
        mt_counts = [int(r["threads_used"]) for r in rows_for_csv
                     if r["backend"] == "mt" and r.get("op") == op and str(r.get("threads_used","")).isdigit()]
        if mt_counts:
            avg = sum(mt_counts) / len(mt_counts)
            uniq = sorted(set(mt_counts))
            print(f"[threads] {op}: CPU MT avg={avg:.2f}, unique={uniq}")
        else:
            print(f"[threads] {op}: no MT runs (or no thread info parsed)")

if __name__ == "__main__":
    main()