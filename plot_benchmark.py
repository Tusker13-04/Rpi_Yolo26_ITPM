#!/usr/bin/env python3
"""
plot_benchmark.py  —  Generate inference latency plot from benchmark CSV.
Usage:
    python3 plot_benchmark.py --csv results.csv --out benchmark_plot.pdf
"""

import argparse, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Publication style ────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          9,
    "axes.linewidth":     0.6,
    "axes.edgecolor":     "#333",
    "axes.facecolor":     "white",
    "figure.facecolor":   "white",
    "grid.color":         "#ddd",
    "grid.linewidth":     0.4,
    "xtick.major.width":  0.5,
    "ytick.major.width":  0.5,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "legend.frameon":     False,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",  required=True, help="Path to results CSV")
    p.add_argument("--out",  default="benchmark_plot.pdf", help="Output file (.pdf or .png)")
    p.add_argument("--title", default="YOLOE-26s — Inference latency per image (RPi 4)")
    return p.parse_args()

def main():
    args = parse_args()
    rows = []
    with open(args.csv, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                try:
                    rows.append((row[0], float(row[1]), int(row[2]) if len(row) > 2 else 0))
                except ValueError:
                    continue  # skip header

    if not rows:
        print("[plot_benchmark] No data found in CSV. Exiting.")
        return

    names   = [r[0] for r in rows]
    times   = np.array([r[1] for r in rows])
    dets    = np.array([r[2] for r in rows])

    mean_t  = np.mean(times)
    std_t   = np.std(times)
    min_t   = np.min(times)
    max_t   = np.max(times)
    fps_est = 1000.0 / mean_t

    short_names = [Path(n).stem.replace("_yoloe", "")[:20] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, max(3.5, len(rows) * 0.4 + 1.5)),
                                   gridspec_kw={"width_ratios": [3, 1.4]})

    y_pos = np.arange(len(names))
    bars  = ax1.barh(y_pos, times, height=0.55, color="#2563eb", alpha=0.82)
    ax1.axvline(mean_t, color="#dc2626", linewidth=0.9, linestyle="--", label=f"Mean = {mean_t:.0f} ms")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(short_names, fontsize=8)
    ax1.set_xlabel("Inference latency (ms)", fontsize=9)
    ax1.set_title(args.title, fontsize=9, pad=8, loc="left")
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.grid(axis="x", which="major")
    ax1.legend(fontsize=8)

    for bar, val in zip(bars, times):
        ax1.text(bar.get_width() + max(times) * 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.0f} ms", va="center", ha="left", fontsize=7.5, color="#333")

    ax2.axis("off")
    stats = [
        ("Images",      f"{len(rows)}"),
        ("Mean",        f"{mean_t:.1f} ms"),
        ("Std dev",     f"{std_t:.1f} ms"),
        ("Min",         f"{min_t:.1f} ms"),
        ("Max",         f"{max_t:.1f} ms"),
        ("Est. FPS",    f"{fps_est:.2f}"),
        ("Total dets.", f"{int(dets.sum())}"),
    ]
    col_labels = ["Metric", "Value"]
    cell_text  = [[k, v] for k, v in stats]
    tbl = ax2.table(cellText=cell_text, colLabels=col_labels,
                    loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#ccc")
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_facecolor("#f0f0f0")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("white")
    ax2.set_title("Summary statistics", fontsize=9, pad=8, loc="left")

    plt.tight_layout(pad=1.2)

    out = Path(args.out)
    plt.savefig(str(out), dpi=300, bbox_inches="tight")
    if out.suffix.lower() == ".pdf":
        plt.savefig(str(out.with_suffix(".png")), dpi=300, bbox_inches="tight")
    print(f"[plot_benchmark] Saved: {out}")

if __name__ == "__main__":
    main()
