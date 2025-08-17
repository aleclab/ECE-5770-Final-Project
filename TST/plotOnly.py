# plotOnly.py — Averages + per-op scatters (linear & log-Y) + per-op line graphs (linear & log-Y),
# incl. NPP, with Sobel generic "CUDA" suppressed when variants present, extra-wide figures,
# and DISTINCT COLORS for CUDA variants v0..v3.

import argparse
import csv
from pathlib import Path
from collections import defaultdict
import statistics
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- figure sizing ----------
AVG_DPI         = 130
SCATTER_DPI     = 150
LINE_DPI        = 150

AVG_WIDTH_IN    = 7.5
SCATTER_SIZE_IN = (22, 6.5)   # very wide
LINE_SIZE_IN    = (22, 6.5)   # very wide

# ---------- Display/style ----------
BACKEND_DISPLAY = {
    "cpu": "CPU",
    "mt": "CPU MT",
    "cuda": "CUDA",
    "cuda_v0": "CUDA v0",
    "cuda_v1": "CUDA v1",
    "cuda_v2": "CUDA v2",
    "cuda_v3": "CUDA v3",
    "npp": "NPP",
}

# DISTINCT colors per CUDA variant; avoid collisions with CPU/MT/NPP
# Palette mostly from matplotlib tab10: blue, purple, red, cyan, plus distinct for generic CUDA
BACKEND_COLOR = {
    "cpu":     "#7f7f7f",  # gray
    "mt":      "#2ca02c",  # green
    "npp":     "#ff7f0e",  # orange
    "cuda":    "#8c564b",  # brown (generic CUDA when it appears for non-Sobel ops)
    "cuda_v0": "#1f77b4",  # blue
    "cuda_v1": "#9467bd",  # purple
    "cuda_v2": "#d62728",  # red
    "cuda_v3": "#17becf",  # cyan
}

BACKEND_MARKER = {
    "cpu": "s",
    "mt": "D",
    "cuda": "o",
    "cuda_v0": "o",
    "cuda_v1": "o",
    "cuda_v2": "o",
    "cuda_v3": "o",
    "npp": "^",
}
BACKEND_ORDER = ["cpu", "mt", "cuda_v0", "cuda_v1", "cuda_v2", "cuda_v3", "cuda", "npp"]
OPS_ALLOWED = ("sobel", "sharpen", "gaussian")

def safe_float(x):
    try: return float(x)
    except: return None

def safe_int(x):
    try: return int(x)
    except: return None

def backend_key_for_row(row: dict) -> str:
    """Map CSV row to display backend key; split CUDA by variant if present."""
    b = (row.get("backend") or "").strip().lower()
    if b == "cuda":
        cv = row.get("cuda_variant")
        if cv is None or str(cv).strip() == "":
            return "cuda"
        try:
            return f"cuda_v{int(cv)}"
        except:
            return "cuda"
    return b

def load_rows(csv_path: Path, ops_filter=None):
    rows = []
    with open(csv_path, "r", newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            op = (r.get("op") or "").strip().lower()
            if ops_filter and op not in ops_filter:
                continue
            width  = safe_int(r.get("width"))
            height = safe_int(r.get("height"))
            area   = safe_int(r.get("area"))
            if area is None and width is not None and height is not None:
                area = width * height
            cuda_variant = r.get("cuda_variant")
            cv_parsed = None
            if cuda_variant not in (None, "", "-"):
                try: cv_parsed = int(cuda_variant)
                except: cv_parsed = None

            row = {
                "index": safe_int(r.get("index")),
                "image": r.get("image"),
                "op": op,
                "backend": (r.get("backend") or "").strip().lower(),
                "cuda_variant": cv_parsed,
                "threads_used": safe_int(r.get("threads_used")),
                "time_ms": safe_float(r.get("time_ms")),
                "correct": str(r.get("correct","")).strip().lower() in ("1","true","yes"),
                "width": width,
                "height": height,
                "area": area,
            }
            row["bk_key"] = backend_key_for_row(row)
            rows.append(row)
    return rows

def order_backends(keys):
    present = sorted(set(keys))
    present.sort(key=lambda k: (BACKEND_ORDER.index(k) if k in BACKEND_ORDER else 999, k))
    return present

def sobel_suppress_generic_cuda(rows_for_op):
    """If any CUDA variants exist for Sobel, drop generic 'cuda' entries to avoid a 5th series."""
    has_variants = any(r["backend"]=="cuda" and r.get("cuda_variant") is not None for r in rows_for_op)
    if has_variants:
        return [r for r in rows_for_op if not (r["backend"]=="cuda" and r.get("cuda_variant") is None)]
    return rows_for_op

# ---------- Average plot ----------
def plot_avg_per_backend(op: str, rows, out_path: Path):
    rows_op = [r for r in rows if r["op"]==op and r["time_ms"] is not None]
    if not rows_op:
        print(f"[plot] skip {op} (no rows)")
        return
    if op == "sobel":
        rows_op = sobel_suppress_generic_cuda(rows_op)

    times_by_bk = defaultdict(list)
    for r in rows_op:
        times_by_bk[r["bk_key"]].append(r["time_ms"])

    bk_keys = order_backends(times_by_bk.keys())
    means = [statistics.mean(times_by_bk[k]) for k in bk_keys]
    ns    = [len(times_by_bk[k]) for k in bk_keys]
    labels = [BACKEND_DISPLAY.get(k, k.upper()) for k in bk_keys]
    colors = [BACKEND_COLOR.get(k, "#444444") for k in bk_keys]

    height_per_bar = 0.6
    fig_height = max(2.8, 1.1 + height_per_bar * len(bk_keys))
    fig, ax = plt.subplots(figsize=(AVG_WIDTH_IN, fig_height), dpi=AVG_DPI)

    y = np.arange(len(bk_keys))
    ax.barh(y, means, color=colors, edgecolor="black", alpha=0.9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Average time (ms)")
    ax.set_title(f"{op.capitalize()} — Average time by backend")

    xmax = max(means) if means else 1.0
    ax.set_xlim(0, xmax * 1.25)
    for yi, (m, n) in enumerate(zip(means, ns)):
        ax.text(m * 1.01, yi, f"{m:.2f} ms  (N={n})", va="center", fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")

# ---------- Per-op scatter (helper) ----------
def _plot_scatter(rows_op, out_path: Path, title: str, ylog: bool):
    """Generic scatter writer; assumes rows_op already filtered and sobel-suppressed."""
    fig, ax = plt.subplots(figsize=SCATTER_SIZE_IN, dpi=SCATTER_DPI)
    ax.set_title(title)
    ax.set_xlabel("Image area (pixels)")
    ax.set_ylabel("Time (ms)")

    by_bk = defaultdict(list)
    for r in rows_op:
        by_bk[r["bk_key"]].append(r)

    for bk in order_backends(by_bk.keys()):
        pts = by_bk[bk]
        xs = [p["area"] for p in pts if p["area"] is not None and p["time_ms"] is not None]
        ys = [p["time_ms"] for p in pts if p["area"] is not None and p["time_ms"] is not None]
        if not xs:
            continue
        ax.scatter(
            xs, ys,
            s=28,
            marker=BACKEND_MARKER.get(bk, "o"),
            color=BACKEND_COLOR.get(bk, "#444444"),
            alpha=0.95,
            label=BACKEND_DISPLAY.get(bk, bk.upper()),
            edgecolors="none",
        )

    # Log-X if areas span >10x
    finite_areas = [r["area"] for r in rows_op if r["area"] not in (None, 0)]
    if finite_areas:
        amin, amax = min(finite_areas), max(finite_areas)
        if amin > 0 and (amax / max(amin, 1) >= 10):
            ax.set_xscale("log")

    # Optional log-Y version
    if ylog:
        pos_times = [r["time_ms"] for r in rows_op if r["time_ms"] and r["time_ms"] > 0]
        if pos_times:
            ax.set_yscale("log")

    ax.grid(True, which="both", linestyle=":", alpha=0.35)
    ax.legend(loc="best", frameon=True, fontsize=10, ncols=1)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"[scatter] wrote {out_path}")

# ---------- Per-op line (helper) ----------
def _plot_line(rows_op, out_path: Path, title: str, ylog: bool):
    """Line graph: per backend, connect points sorted by area."""
    fig, ax = plt.subplots(figsize=LINE_SIZE_IN, dpi=LINE_DPI)
    ax.set_title(title)
    ax.set_xlabel("Image area (pixels)")
    ax.set_ylabel("Time (ms)")

    by_bk = defaultdict(list)
    for r in rows_op:
        if r["area"] is None or r["time_ms"] is None:
            continue
        by_bk[r["bk_key"]].append((r["area"], r["time_ms"]))

    any_series = False
    for bk in order_backends(by_bk.keys()):
        pts = by_bk[bk]
        if not pts:
            continue
        pts.sort(key=lambda t: t[0])  # sort by area
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        any_series = True
        ax.plot(
            xs, ys,
            marker=BACKEND_MARKER.get(bk, "o"),
            linewidth=1.8,
            markersize=4.5,
            color=BACKEND_COLOR.get(bk, "#444444"),
            alpha=0.95,
            label=BACKEND_DISPLAY.get(bk, bk.upper()),
        )

    if not any_series:
        plt.close(fig)
        return

    # Log-X if areas span >10x
    finite_areas = [r["area"] for r in rows_op if r["area"] not in (None, 0)]
    if finite_areas:
        amin, amax = min(finite_areas), max(finite_areas)
        if amin > 0 and (amax / max(amin, 1) >= 10):
            ax.set_xscale("log")

    # Optional log-Y
    if ylog:
        pos_times = [r["time_ms"] for r in rows_op if r["time_ms"] and r["time_ms"] > 0]
        if pos_times:
            ax.set_yscale("log")

    ax.grid(True, which="both", linestyle=":", alpha=0.35)
    ax.legend(loc="best", frameon=True, fontsize=10, ncols=1)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"[line] wrote {out_path}")

# ---------- Per-op (public) ----------
def plot_scatter_and_lines_per_op(rows, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for op in OPS_ALLOWED:
        rows_op = [r for r in rows if r["op"]==op and r["time_ms"] is not None and r["area"] is not None]
        if not rows_op:
            print(f"[scatter/line] skip {op} (no rows)")
            continue
        if op == "sobel":
            rows_op = sobel_suppress_generic_cuda(rows_op)

        # Scatter: linear & log-Y
        _plot_scatter(
            rows_op,
            out_dir / f"{op}_time_vs_area_linear.png",
            f"{op.capitalize()} — Time vs. Pixels (linear Y)",
            ylog=False
        )
        _plot_scatter(
            rows_op,
            out_dir / f"{op}_time_vs_area_logy.png",
            f"{op.capitalize()} — Time vs. Pixels (log Y)",
            ylog=True
        )

        # Line: linear & log-Y
        _plot_line(
            rows_op,
            out_dir / f"{op}_time_vs_area_line_linear.png",
            f"{op.capitalize()} — Time vs. Pixels (line, linear Y)",
            ylog=False
        )
        _plot_line(
            rows_op,
            out_dir / f"{op}_time_vs_area_line_logy.png",
            f"{op.capitalize()} — Time vs. Pixels (line, log Y)",
            ylog=True
        )

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Plot averages and per-op scatters/lines from CSV (incl. NPP).")
    ap.add_argument("--csv", required=True, help="Input CSV file.")
    ap.add_argument("--out-plots", required=True, help="Directory for avg plots (and scatters/lines if --scatter-out not set).")
    ap.add_argument("--ops", nargs="+", default=list(OPS_ALLOWED), help="Subset of ops: sobel sharpen gaussian")
    ap.add_argument("--scatter", action="store_true", help="Also write scatters/lines per op (linear & log-Y).")
    ap.add_argument("--scatter-out", default=None, help="Optional different directory for scatter/line plots.")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    plots_dir = Path(args.out_plots)
    scat_dir = Path(args.scatter_out) if args.scatter_out else plots_dir

    if not csv_path.is_file():
        raise SystemExit(f"[err] CSV not found: {csv_path}")
    plots_dir.mkdir(parents=True, exist_ok=True)
    scat_dir.mkdir(parents=True, exist_ok=True)

    ops_filter = [o for o in args.ops if o in OPS_ALLOWED]
    rows = load_rows(csv_path, ops_filter=ops_filter)
    if not rows:
        print("[warn] no rows to plot.")
        return

    # Average bars
    for op in ops_filter:
        plot_avg_per_backend(op, rows, plots_dir / f"{op}_avg_by_backend.png")

    # Scatters + Lines
    if args.scatter:
        plot_scatter_and_lines_per_op(rows, scat_dir)

if __name__ == "__main__":
    main()
