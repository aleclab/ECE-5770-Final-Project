import argparse, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Match your plotting conventions
OP_MARKER = {"sobel":"o","sharpen":"s","gaussian":"^"}
BACKEND_COLOR = {"cuda":"C0","cpu":"C1","mt":"C2"}
BACKEND_DISPLAY = {"cuda":"CUDA","cpu":"CPU","mt":"CPU MT"}

def read_rows(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def coerce_float(x):
    try:
        return float(x)
    except Exception:
        return None

def get_present_sets(rows):
    ops = sorted(set(r.get("op","") for r in rows if r.get("op")))
    bks = sorted(set(r.get("backend","") for r in rows if r.get("backend")))
    return ops, bks

def plot_avg_by_backend(rows, ops, backends, out_dir: Path, title_suffix=""):
    out_dir.mkdir(parents=True, exist_ok=True)
    for op in ops:
        # collect series per backend
        labels, means, stdevs, ns = [], [], [], []
        for bk in backends:
            vals = []
            for r in rows:
                if r.get("op") != op or r.get("backend") != bk:
                    continue
                t = coerce_float(r.get("time_ms",""))
                if t is not None:
                    vals.append(t)
            if not vals:
                continue
            labels.append(BACKEND_DISPLAY.get(bk, bk))
            means.append(float(np.mean(vals)))
            stdevs.append(float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0)
            ns.append(len(vals))
        if not labels:
            print(f"[plot] no data for op={op}")
            continue

        x = np.arange(len(labels))
        plt.figure()
        plt.bar(x, means, yerr=stdevs, capsize=4)
        plt.xticks(x, labels)
        plt.ylabel("kernel time (ms)")
        ttl = f"{op}: average kernel time across images"
        if title_suffix: ttl += f" ({title_suffix})"
        plt.title(ttl)
        for xi, (m, n) in zip(x, zip(means, ns)):
            plt.text(xi, m, f"{m:.2f}\nN={n}", ha="center", va="bottom")
        plt.tight_layout()
        out_path = out_dir / f"{op}_avg_by_backend.png"
        plt.savefig(out_path)
        plt.close()
        print(f"[plot] wrote {out_path}")

def plot_scatter(rows, out_path: Path, ops, backends, max_points=50000, title_suffix=""):
    # group by (backend, op)
    groups = {}
    total = 0
    for r in rows:
        op = r.get("op","")
        bk = r.get("backend","")
        if op not in ops or bk not in backends:
            continue
        t = coerce_float(r.get("time_ms",""))
        if t is None or t <= 0: continue
        area = None
        if "area" in r and str(r["area"]).strip():
            try: area = int(r["area"])
            except: area = None
        if area is None:
            w = coerce_float(r.get("width",""))
            h = coerce_float(r.get("height",""))
            if w is None or h is None: continue
            area = int(w*h)
        if area <= 0: continue
        groups.setdefault((bk,op), []).append((area, t))
        total += 1
    if total == 0:
        print("[scatter] no datapoints to plot")
        return

    # downsample if huge
    if total > max_points:
        new_groups, acc = {}, 0
        for key, pts in groups.items():
            n = len(pts)
            quota = max(1, int(round(max_points * (n / total))))
            if n <= quota:
                new_groups[key] = pts
                acc += n
            else:
                idxs = np.linspace(0, n-1, quota).astype(int)
                new_groups[key] = [pts[i] for i in idxs]
                acc += quota
        groups = new_groups
        total = acc
        print(f"[scatter] downsampled to ~{total} points for faster plotting")

    plt.figure(figsize=(8,6))
    for (bk,op), pts in groups.items():
        if not pts: continue
        arr = np.asarray(pts, dtype=float)
        xs, ys = arr[:,0], arr[:,1]
        plt.scatter(xs, ys, color=BACKEND_COLOR.get(bk,"C7"),
                    marker=OP_MARKER.get(op,"x"), alpha=0.5, s=16, linewidths=0)

    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("image area (pixels)")
    plt.ylabel("kernel time (ms, log scale)")
    ttl = "Kernel time vs image area (color=backend, marker=op)"
    if title_suffix: ttl += f" ({title_suffix})"
    plt.title(ttl)

    backend_handles = [plt.Line2D([0],[0], color=BACKEND_COLOR.get(bk,"C7"),
                         marker='o', linestyle='None', label=BACKEND_DISPLAY.get(bk,bk))
                       for bk in backends]
    op_handles = [plt.Line2D([0],[0], color="black", marker=OP_MARKER.get(op,"x"),
                       linestyle='None', label=op) for op in ops]
    leg1 = plt.legend(handles=backend_handles, title="Backend", loc="upper left")
    plt.gca().add_artist(leg1)
    plt.legend(handles=op_handles, title="Op", loc="lower right")
    plt.tight_layout()

    if out_path.exists() and out_path.is_dir():
        out_path = out_path / "time_vs_area.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path); plt.close()
    print(f"[plot] wrote {out_path}")

def summarize_threads(rows, ops):
    for op in ops:
        mt = [int(r["threads_used"]) for r in rows
              if r.get("backend")=="mt" and r.get("op")==op and str(r.get("threads_used","")).isdigit()]
        if mt:
            avg = sum(mt)/len(mt)
            uniq = sorted(set(mt))
            print(f"[threads] {op}: CPU MT avg={avg:.2f}, unique={uniq}")
        else:
            print(f"[threads] {op}: no MT rows in CSV")

def main():
    ap = argparse.ArgumentParser(description="Plot averages & scatter from an existing CSV (no re-run).")
    ap.add_argument("--csv", required=True, help="Path to datum.csv produced by the runner")
    ap.add_argument("--out-plots", default="Graphs", help="Directory for average plots")
    ap.add_argument("--scatter", action="store_true", help="Also emit scatter plot")
    ap.add_argument("--scatter-out", default="Graphs/Scatters/time_vs_area.png",
                    help="Path (or directory) for scatter PNG")
    ap.add_argument("--ops", nargs="*", default=None, help="Subset of ops to include")
    ap.add_argument("--backends", nargs="*", default=None, help="Subset of backends to include")
    ap.add_argument("--max-points", type=int, default=50000, help="Cap total scatter points")
    ap.add_argument("--title-suffix", default="", help="Optional text to add to plot titles")
    args = ap.parse_args()

    rows = read_rows(args.csv)
    if not rows:
        print("[err] CSV empty or unreadable")
        return

    present_ops, present_bks = get_present_sets(rows)
    ops = args.ops if args.ops else present_ops
    backends = args.backends if args.backends else [bk for bk in ["cuda","cpu","mt"] if bk in present_bks]
    if not ops or not backends:
        print("[err] nothing to plot (check --ops/--backends)")
        return

    plots_dir = Path(args.out_plots)
    plot_avg_by_backend(rows, ops, backends, plots_dir, args.title_suffix)

    if args.scatter:
        scatter_out = Path(args.scatter_out)
        plot_scatter(rows, scatter_out, ops, backends, args.max_points, args.title_suffix)

    summarize_threads(rows, ops)

if __name__ == "__main__":
    main()