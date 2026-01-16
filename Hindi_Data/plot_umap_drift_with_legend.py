#!/usr/bin/env python3
"""
Regenerate UMAP drift arrows with an explicit legend for original (Hindi/Roman Hindi) vs translated English.

Usage:
  python plot_umap_drift_with_legend.py \
      --coords min-check/umap_before_after_coords.csv \
      --out min-check/umap_drift_arrows_with_legend.png
"""
import argparse
import csv
from typing import List, Tuple

import matplotlib.pyplot as plt


def load_coords(path: str) -> List[Tuple[float, float, float, float, str]]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = [
            "umap_x_before",
            "umap_y_before",
            "umap_x_after",
            "umap_y_after",
            "script",
        ]
        for col in required:
            if col not in reader.fieldnames:
                raise ValueError(f"Missing column '{col}' in {path}")
        for row in reader:
            try:
                xb = float(row["umap_x_before"])
                yb = float(row["umap_y_before"])
                xa = float(row["umap_x_after"])
                ya = float(row["umap_y_after"])
                script = row.get("script", "")
            except Exception as exc:
                raise ValueError(f"Bad row in {path}: {row}") from exc
            rows.append((xb, yb, xa, ya, script))
    return rows


def plot(coords: List[Tuple[float, float, float, float, str]], out: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    xs_b, ys_b, xs_a, ys_a = [], [], [], []
    for xb, yb, xa, ya, _ in coords:
        xs_b.append(xb)
        ys_b.append(yb)
        xs_a.append(xa)
        ys_a.append(ya)
        ax.arrow(
            xb,
            yb,
            xa - xb,
            ya - yb,
            length_includes_head=True,
            head_width=0.08,
            head_length=0.1,
            color="gray",
            alpha=0.35,
            linewidth=0.6,
        )

    before_pts = ax.scatter(xs_b, ys_b, s=12, c="#1f77b4", alpha=0.85, label="Original (Hindi/Roman)")
    after_pts = ax.scatter(xs_a, ys_a, s=12, c="#ff7f0e", alpha=0.85, label="Translated (English)")

    ax.set_xlabel("UMAP x")
    ax.set_ylabel("UMAP y")
    ax.set_title("Translation Drift: Original → English (UMAP space)")
    # Legend explicitly distinguishes Hindi/Roman vs English points.
    ax.legend(handles=[before_pts, after_pts], loc="best", frameon=True)
    ax.grid(True, linewidth=0.3, alpha=0.4)

    plt.tight_layout()
    fig.savefig(out)
    print(f"[OK] Wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", default="min-check/umap_before_after_coords.csv", help="CSV with before/after UMAP coords")
    ap.add_argument("--out", default="umap_drift_arrows_with_legend.png", help="Output PNG path")
    args = ap.parse_args()

    coords = load_coords(args.coords)
    if not coords:
        raise SystemExit(f"No rows found in {args.coords}")
    plot(coords, args.out)


if __name__ == "__main__":
    main()
