#!/usr/bin/env python3
"""
Analyze translation drift and visualize low-cosine cases.

Outputs:
  - CSV of rows below a cosine similarity threshold (default 0.9).
  - Line plot (two subplots) showing UMAP X and Y coordinates for originals vs translations,
    ordered by ascending cosine similarity (worst first).

Usage:
  MPLBACKEND=Agg MPLCONFIGDIR=/tmp ./pdxvenv/bin/python analyze_drift.py \
      --coords min-check/umap_before_after_coords.csv \
      --thr 0.9 \
      --out-prefix min-check/umap_lowcos
"""
import argparse
import csv
import math
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


Row = Tuple[int, float, float, float, float, float, str, str]
# (orig_index, cos, xb, yb, xa, ya, script, tweet_snip)


def load_rows(path: Path) -> List[Row]:
    rows: List[Row] = []
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        required = [
            "orig_index",
            "cosine_before_after",
            "umap_x_before",
            "umap_y_before",
            "umap_x_after",
            "umap_y_after",
            "script",
            "tweet",
            "en_text",
        ]
        for col in required:
            if col not in r.fieldnames:
                raise ValueError(f"Missing column '{col}' in {path}")
        for row in r:
            try:
                rows.append(
                    (
                        int(row["orig_index"]),
                        float(row["cosine_before_after"]),
                        float(row["umap_x_before"]),
                        float(row["umap_y_before"]),
                        float(row["umap_x_after"]),
                        float(row["umap_y_after"]),
                        row.get("script", ""),
                        row.get("tweet", "")[:160],
                    )
                )
            except Exception as exc:
                raise ValueError(f"Bad row in {path}: {row}") from exc
    return rows


def write_lows(rows: List[Row], thr: float, out_csv: Path) -> List[Row]:
    lows = [r for r in rows if r[1] < thr]
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "orig_index",
                "cosine_before_after",
                "umap_x_before",
                "umap_y_before",
                "umap_x_after",
                "umap_y_after",
                "script",
                "tweet_snip",
            ]
        )
        for r in lows:
            w.writerow(r)
    return lows


def plot_lines(lows: List[Row], out_png: Path) -> None:
    if not lows:
        print("[warn] no rows below threshold; skipping plot")
        return
    # Order by cosine ascending (worst drift first)
    lows = sorted(lows, key=lambda r: r[1])
    x_axis = list(range(len(lows)))
    xb = [r[2] for r in lows]
    yb = [r[3] for r in lows]
    xa = [r[4] for r in lows]
    ya = [r[5] for r in lows]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), dpi=150, sharex=True)

    axes[0].plot(x_axis, xb, label="Original UMAP x", color="#1f77b4")
    axes[0].plot(x_axis, xa, label="Translated UMAP x", color="#ff7f0e")
    axes[0].set_ylabel("UMAP x")
    axes[0].legend()
    axes[0].grid(True, linewidth=0.3, alpha=0.5)

    axes[1].plot(x_axis, yb, label="Original UMAP y", color="#1f77b4")
    axes[1].plot(x_axis, ya, label="Translated UMAP y", color="#ff7f0e")
    axes[1].set_ylabel("UMAP y")
    axes[1].set_xlabel("Tweet (ordered by cosine similarity, worst → best)")
    axes[1].legend()
    axes[1].grid(True, linewidth=0.3, alpha=0.5)

    fig.suptitle("Low-cosine translations (< threshold): UMAP coordinates by tweet")
    plt.tight_layout()
    fig.savefig(out_png)
    print(f"[OK] wrote plot {out_png}")


def plot_xy_scatter(lows: List[Row], out_png: Path) -> None:
    if not lows:
        print("[warn] no rows below threshold; skipping XY scatter plot")
        return
    lows = sorted(lows, key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    for idx, r in enumerate(lows):
        xb, yb, xa, ya = r[2], r[3], r[4], r[5]
        ax.arrow(
            xb,
            yb,
            xa - xb,
            ya - yb,
            length_includes_head=True,
            head_width=0.06,
            head_length=0.08,
            color="gray",
            alpha=0.35,
            linewidth=0.6,
            zorder=1,
        )
    before_pts = ax.scatter([r[2] for r in lows], [r[3] for r in lows], s=14, c="#1f77b4", alpha=0.85, label="Original (UMAP)")
    after_pts = ax.scatter([r[4] for r in lows], [r[5] for r in lows], s=14, c="#ff7f0e", alpha=0.85, label="Translated (UMAP)")
    ax.set_xlabel("UMAP x")
    ax.set_ylabel("UMAP y")
    ax.set_title("Low-cosine translations: UMAP XY (original vs translated)")
    ax.legend(handles=[before_pts, after_pts], loc="best", frameon=True)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    plt.tight_layout()
    fig.savefig(out_png)
    print(f"[OK] wrote plot {out_png}")


def summarize(rows: List[Row]) -> None:
    sims = [r[1] for r in rows]
    sims_sorted = sorted(sims)
    n = len(sims_sorted)
    def pct(p):
        return sims_sorted[int(p * n)]
    print(f"[summary] rows={n} min={sims_sorted[0]:.4f} p10={pct(0.10):.4f} median={pct(0.50):.4f} p90={pct(0.90):.4f} max={sims_sorted[-1]:.4f}")

    by_script = {}
    for r in rows:
        by_script.setdefault(r[6], []).append(r[1])
    for s, vals in by_script.items():
        vals = sorted(vals)
        m = len(vals)
        def p(pct):
            return vals[int(pct * m)]
        print(f"  [script={s}] n={m} min={vals[0]:.4f} p10={p(0.10):.4f} median={p(0.50):.4f} p90={p(0.90):.4f} max={vals[-1]:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", required=True, help="Path to umap_before_after_coords CSV")
    ap.add_argument("--thr", type=float, default=0.9, help="Cosine similarity threshold for low-drift selection")
    ap.add_argument("--out-prefix", default="umap_lowcos", help="Prefix for outputs (CSV and PNG)")
    args = ap.parse_args()

    coords_path = Path(args.coords)
    rows = load_rows(coords_path)
    summarize(rows)

    out_csv = Path(f"{args.out_prefix}_below_{args.thr:.2f}.csv")
    lows = write_lows(rows, args.thr, out_csv)
    print(f"[OK] wrote {len(lows)} rows below {args.thr} to {out_csv}")

    out_png = Path(f"{args.out_prefix}_below_{args.thr:.2f}.png")
    plot_lines(lows, out_png)

    out_xy = Path(f"{args.out_prefix}_below_{args.thr:.2f}_xy.png")
    plot_xy_scatter(lows, out_xy)


if __name__ == "__main__":
    main()
