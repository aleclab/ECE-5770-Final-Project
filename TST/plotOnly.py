# plot_from_csv.py
import argparse
import csv
from pathlib import Path
import numpy as np
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Display / style mappings (mirror main script) ----
BACKEND_DISPLAY = {
    "cuda": "CUDA",
    "cpu":  "CPU",
    "mt":   "CPU MT",
}
BACKEND_COLOR = {
    "cuda": "C0",
    "cpu":  "C1",
    "mt":   "C2",
}
OP_MARKER = {
    "sobel":    "o",
    "sharpen":  "s",
    "gaussian": "^",
}

# CUDA variant labeling (keep in sync with your main script)
CUDA_VARIANT_LABEL = {
    0: "CUDA v0 (original)",
    1: "CUDA v1",
    2: "CUDA v2",
    3: "CUDA v3",
}
CUDA_VARIANT_MARKER = {
    0: "o",  # circle
    1: "^",  # triangle
    2: "s",  # square
    3: "D",  # diamond
}
# distinct colors for variants so they pop, leave CPU/MT as-is
CUDA_VARIANT_COLOR = {
    0: "C0",  # blue
    1: "C3",  # red-ish
    2: "C4",  # purple-ish
    3: "C5",  # brown-ish
}

def backend_sort_key(key: str):
    """Order: cuda, cuda_v0..v3, cpu, mt, then anything else."""
    if key == "cuda":
        return (0, -1)
    if key.startswith("cuda_v"):
        try:
            v = int(key.split("cuda_v", 1)[1])
        except Exception:
            v = 99
        return (1, v)
    if key == "cpu":
        return (10, 0)
    if key == "mt":
        return (11, 0)
    return (99, 0)

def read_rows(csv_path: Path, ops_filter=None):
    rows = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for rec in r:
            op = (rec.get("op") or "").strip().lower()
            if ops_filter and op not in ops_filter:
                continue
            # parse numerics
            def to_float(s):
                try:
                    return float(s)
                except Exception:
                    return None
            def to_int(s):
                try:
                    return int(s)
                except Exception:
                    return None
            time_ms = to_float(rec.get("time_ms", ""))
            width   = to_int(rec.get("width", ""))
            height  = to_int(rec.get("height", ""))
            area    = to_int(rec.get("area", ""))
            threads = to_int(rec.get("threads_used", ""))
            backend = (rec.get("backend") or "").strip().lower()
            cv = rec.get("cuda_variant", "")
            try:
                cuda_variant = int(cv) if str(cv).strip() != "" else None
            except Exception:
                cuda_variant = None
            rows.append({
                "op": op,
                "backend": backend,
                "cuda_variant": cuda_variant,
                "time_ms": time_ms,
                "width": width, "height": height, "area": area,
                "threads_used": threads,
                "image": rec.get("image", ""),
                "index": rec.get("index", ""),
                "correct": rec.get("correct", ""),
            })
    return rows

def plot_avg_per_backend_from_rows(op: str, rows, out_path: Path):
    # group times by label (expand CUDA variants if present)
    times = {}
    for r in rows:
        if r["op"] != op:
            continue
        t = r["time_ms"]
        if t is None:
            continue
        bk = r["backend"]
        if bk == "cuda" and r["cuda_variant"] is not None:
            key = f"cuda_v{r['cuda_variant']}"
            label = CUDA_VARIANT_LABEL.get(r["cuda_variant"], f"CUDA v{r['cuda_variant']}")
        else:
            key = bk
            label = BACKEND_DISPLAY.get(bk, bk)
        times.setdefault(key, {"label": label, "vals": []})
        times[key]["vals"].append(float(t))

    if not times:
        print(f"[plot] no data to plot for {op}")
        return

    # order keys
    keys = sorted(times.keys(), key=backend_sort_key)
    labels = [times[k]["label"] for k in keys]
    means  = [float(np.mean(times[k]["vals"])) for k in keys]
    stdevs = [float(np.std(times[k]["vals"], ddof=1)) if len(times[k]["vals"]) > 1 else 0.0 for k in keys]
    ns     = [len(times[k]["vals"]) for k in keys]

    # dynamic width so long labels don’t overlap
    width_in  = max(10.0, 0.9 * len(labels) + 2.0)
    height_in = 6.0
    plt.figure(figsize=(width_in, height_in))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=stdevs, capsize=4)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("kernel time (ms)")
    plt.title(f"{op}: average kernel time across images (N shown)")
    for xi, (m, n) in zip(x, zip(means, ns)):
        plt.text(xi, m, f"{m:.2f}\nN={n}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[plot] wrote {out_path}")

def plot_scatter_time_vs_area_from_rows(
    rows, out_path: Path,
    max_points_per_group=None,
    jitter_frac=0.0,            # e.g., 0.02 for ±2% multiplicative jitter on x & y (log-safe)
    rng_seed=12345              # set None for non-deterministic jitter
):
    """
    Scatter with stable draw order and optional log-safe jitter.

    Marker (shape) encodes OP: sobel/sharpen/gaussian
    Color encodes backend, EXCEPT CUDA+Sobel variants use distinct variant colors
    Draw order is fixed: CPU -> MT -> CUDA (plain) -> CUDA v0 -> v1 -> v2 -> v3
    """
    # Requires globals: BACKEND_DISPLAY, BACKEND_COLOR, OP_MARKER,
    # CUDA_VARIANT_COLOR, CUDA_VARIANT_LABEL

    # Key we’ll use to store groups keeps semantics so we can sort consistently:
    # key = (kind, ident, color, marker, label)
    #   kind  = "backend" | "cuda_variant"
    #   ident = "cpu"/"mt"/"cuda"  OR  variant int (0..3)
    groups = {}
    seen_ops = set()
    seen_cpu = False
    seen_mt = False
    seen_cuda_plain = False
    seen_cuda_variants = set()

    if rng_seed is not None and jitter_frac > 0.0:
        rng = np.random.default_rng(rng_seed)
    else:
        rng = None

    def _logsafe_jitter(v):
        if jitter_frac <= 0.0 or rng is None:
            return v
        # multiplicative jitter for log axes: v * exp(u), u ~ U(-a, +a), where a ~ jitter_frac
        u = rng.uniform(-jitter_frac, +jitter_frac, size=v.shape)
        return v * np.exp(u)

    total_kept = 0
    for r in rows:
        t = r.get("time_ms", None)
        area = r.get("area", None)
        if t is None or area is None or t <= 0 or area <= 0:
            continue

        bk = (r.get("backend") or "").lower()
        op = (r.get("op") or "").lower()
        marker = OP_MARKER.get(op, "x")   # shape = operation
        seen_ops.add(op)

        if bk == "cuda":
            v = r.get("cuda_variant", None)
            # CUDA Sobel variants get distinct colors (only meaningful for Sobel)
            if v is not None and op == "sobel":
                try:
                    vi = int(v)
                except Exception:
                    vi = None
                if vi is not None and vi in CUDA_VARIANT_COLOR:
                    color = CUDA_VARIANT_COLOR[vi]
                    label = CUDA_VARIANT_LABEL.get(vi, f"CUDA v{vi}")
                    key = ("cuda_variant", vi, color, marker, label)
                    seen_cuda_variants.add(vi)
                else:
                    color = BACKEND_COLOR.get("cuda", "C0")
                    label = "CUDA (no variant)"
                    key = ("backend", "cuda", color, marker, label)
                    seen_cuda_plain = True
            else:
                color = BACKEND_COLOR.get("cuda", "C0")
                label = "CUDA (no variant)"
                key = ("backend", "cuda", color, marker, label)
                seen_cuda_plain = True
        elif bk == "cpu":
            color = BACKEND_COLOR.get("cpu", "C1")
            label = BACKEND_DISPLAY.get("cpu", "CPU")
            key = ("backend", "cpu", color, marker, label)
            seen_cpu = True
        elif bk == "mt":
            color = BACKEND_COLOR.get("mt", "C2")
            label = BACKEND_DISPLAY.get("mt", "CPU MT")
            key = ("backend", "mt", color, marker, label)
            seen_mt = True
        else:
            color = "C7"
            label = bk or "unknown"
            key = ("backend", bk, color, marker, label)

        g = groups.setdefault(key, {"x": [], "y": []})
        g["x"].append(float(area))
        g["y"].append(float(t))
        total_kept += 1

    if not groups:
        print("[scatter] no datapoints to plot")
        return

    # Fixed, stable draw order
    def group_sort_key(key):
        kind, ident, color, marker, label = key
        if kind == "backend":
            if ident == "cpu":   pri = (0, 0)
            elif ident == "mt":  pri = (1, 0)
            elif ident == "cuda":pri = (2, 0)
            else:                pri = (9, 0)
        elif kind == "cuda_variant":
            # draw variants on top, in numeric order
            try:
                pri = (3, int(ident))
            except Exception:
                pri = (3, 99)
        else:
            pri = (9, 0)
        return pri

    # Plot
    plt.figure(figsize=(10, 7))
    total_plotted = 0

    # Sort groups and assign zorder consistently with sort priority
    sorted_items = sorted(groups.items(), key=lambda kv: group_sort_key(kv[0]))
    for order_idx, (key, data) in enumerate(sorted_items):
        kind, ident, color, marker, label = key
        xs = np.asarray(data["x"], dtype=float)
        ys = np.asarray(data["y"], dtype=float)

        # Optional downsample
        if max_points_per_group is not None and xs.size > max_points_per_group:
            idx = np.linspace(0, xs.size - 1, max_points_per_group).astype(int)
            xs = xs[idx]; ys = ys[idx]

        # Apply reproducible log-safe jitter
        xs = _logsafe_jitter(xs)
        ys = _logsafe_jitter(ys)

        # zorder strictly increases with draw priority so later groups are on top
        z = 10 + order_idx
        plt.scatter(xs, ys, c=color, marker=marker, alpha=0.55, s=18,
                    linewidths=0, rasterized=True, zorder=z)
        total_plotted += xs.size

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("image area (width × height, pixels)")
    plt.ylabel("kernel time (ms, log scale)")
    plt.title("Kernel time vs image area")

    # Legends
    backend_handles = []
    if seen_cpu:
        backend_handles.append(
            plt.Line2D([0],[0], color=BACKEND_COLOR.get("cpu","C1"), marker='o',
                       linestyle='None', label=BACKEND_DISPLAY.get("cpu","CPU"))
        )
    if seen_mt:
        backend_handles.append(
            plt.Line2D([0],[0], color=BACKEND_COLOR.get("mt","C2"), marker='o',
                       linestyle='None', label=BACKEND_DISPLAY.get("mt","CPU MT"))
        )
    if backend_handles:
        leg1 = plt.legend(handles=backend_handles, title="CPU backends (color)", loc="upper left")
        plt.gca().add_artist(leg1)

    op_handles = []
    for op in ("sobel", "sharpen", "gaussian"):
        if op in seen_ops:
            op_handles.append(
                plt.Line2D([0],[0], color="black", marker=OP_MARKER.get(op,"x"),
                           linestyle='None', label=op)
            )
    if op_handles:
        leg2 = plt.legend(handles=op_handles, title="Ops (marker)", loc="upper center")
        plt.gca().add_artist(leg2)

    variant_handles = []
    if seen_cuda_plain:
        variant_handles.append(
            plt.Line2D([0],[0], color=BACKEND_COLOR.get("cuda","C0"), marker='o',
                       linestyle='None', label="CUDA (no variant)")
        )
    for v in sorted(seen_cuda_variants):
        variant_handles.append(
            plt.Line2D([0],[0], color=CUDA_VARIANT_COLOR.get(v, "C0"), marker='o',
                       linestyle='None', label=CUDA_VARIANT_LABEL.get(v, f"CUDA v{v}"))
        )
    if variant_handles:
        plt.legend(handles=variant_handles, title="CUDA Sobel variants (color)", loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[scatter] plotted {total_plotted} points across {len(groups)} groups")
    print(f"[plot] wrote {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Plot averages and scatter from benchmarking CSV.")
    ap.add_argument("--csv", required=True, help="Path to CSV produced by test script")
    ap.add_argument("--out-plots", default=".", help="Directory to save plots")
    ap.add_argument("--ops", nargs="+", default=["gaussian","sobel","sharpen"],
                    help="Subset of ops to plot")
    ap.add_argument("--scatter", action="store_true", help="Also make the time-vs-area scatter plot")
    ap.add_argument("--scatter-out", default=None,
                    help="Output PNG for the scatter plot (default: <plots_dir>/time_vs_area.png)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    plots_dir = Path(args.out_plots)
    plots_dir.mkdir(parents=True, exist_ok=True)

    ops_filter = [o.lower() for o in args.ops]
    rows = read_rows(csv_path, ops_filter=ops_filter)

    # Per-op average charts
    for op in ops_filter:
        out_path = plots_dir / f"{op}_avg_by_backend.png"
        plot_avg_per_backend_from_rows(op, rows, out_path)

    # Scatter (optional)
    if args.scatter:
        scatter_path = Path(args.scatter_out) if args.scatter_out else (plots_dir / "time_vs_area.png")
        if scatter_path.is_dir():
            scatter_path = scatter_path / "time_vs_area.png"
        plot_scatter_time_vs_area_from_rows(rows, scatter_path)

if __name__ == "__main__":
    main()