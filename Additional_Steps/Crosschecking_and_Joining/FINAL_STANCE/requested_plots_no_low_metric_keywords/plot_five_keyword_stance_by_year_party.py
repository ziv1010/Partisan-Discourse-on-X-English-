#!/usr/bin/env python3
"""
Focused stance plot for five requested keywords.

The input summary is produced by make_requested_final_stance_plots.py and
contains stance counts and percentages by keyword, year, and party.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


BASE_DIR = Path(__file__).parent.resolve()
SUMMARY_CSV = BASE_DIR / "keyword_party_stance_percent_by_year_summary.csv"
OUTPUT_CSV = BASE_DIR / "five_keyword_stance_by_year_party_summary.csv"
OUTPUT_PNG = BASE_DIR / "five_keyword_stance_by_year_party.png"
OUTPUT_PDF = BASE_DIR / "five_keyword_stance_by_year_party.pdf"

KEYWORDS = ["bhakts", "democracy", "demonetisation", "modi", "suicides"]
YEARS = ["2020", "2021", "2022", "2023"]
PARTIES = ["pro ruling", "pro opposition"]
STANCES = ["favor", "neutral", "against"]

KEYWORD_LABELS = {
    "bhakts": "Bhakts",
    "democracy": "Democracy",
    "demonetisation": "Demonetisation",
    "modi": "Modi",
    "suicides": "Suicides",
}

PARTY_LABELS = {
    "pro ruling": "Pro Ruling",
    "pro opposition": "Pro Opposition",
}

STANCE_LABELS = {
    "favor": "Favor",
    "neutral": "Neutral",
    "against": "Against",
}

STANCE_COLORS = {
    "favor": "#27ae60",
    "neutral": "#95a5a6",
    "against": "#c0392b",
}


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 9,
        "axes.labelsize": 9.5,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "savefig.facecolor": "white",
    }
)


def load_requested_rows() -> pd.DataFrame:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing summary CSV: {SUMMARY_CSV}")

    df = pd.read_csv(SUMMARY_CSV, dtype={"year": "string"})
    df["keyword"] = df["keyword"].astype(str).str.lower().str.strip()
    df["party"] = df["party"].astype(str).str.lower().str.strip()
    df["year"] = df["year"].astype(str).str.strip()

    filtered = df[
        df["keyword"].isin(KEYWORDS)
        & df["party"].isin(PARTIES)
        & df["year"].isin(YEARS)
    ].copy()

    expected = len(KEYWORDS) * len(PARTIES) * len(YEARS)
    if len(filtered) != expected:
        present = set(zip(filtered["keyword"], filtered["party"], filtered["year"]))
        missing = [
            (keyword, party, year)
            for keyword in KEYWORDS
            for party in PARTIES
            for year in YEARS
            if (keyword, party, year) not in present
        ]
        raise ValueError(f"Expected {expected} rows, found {len(filtered)}. Missing: {missing}")

    filtered["keyword"] = pd.Categorical(filtered["keyword"], categories=KEYWORDS, ordered=True)
    filtered["party"] = pd.Categorical(filtered["party"], categories=PARTIES, ordered=True)
    filtered["year"] = pd.Categorical(filtered["year"], categories=YEARS, ordered=True)
    filtered = filtered.sort_values(["keyword", "party", "year"]).reset_index(drop=True)
    filtered.to_csv(OUTPUT_CSV, index=False)
    return filtered


def annotate_stack(ax: plt.Axes, x_position: float, bottom: float, height: float, value: float) -> None:
    if value < 10:
        return

    ax.text(
        x_position,
        bottom + height / 2,
        f"{value:.0f}%",
        ha="center",
        va="center",
        fontsize=7.4,
        color="black",
        linespacing=1.0,
    )


def plot_rows(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        nrows=len(KEYWORDS),
        ncols=len(PARTIES),
        figsize=(7.6, 9.1),
        sharey=True,
        facecolor="white",
    )

    x = np.arange(len(YEARS))
    width = 0.62

    for row_idx, keyword in enumerate(KEYWORDS):
        for col_idx, party in enumerate(PARTIES):
            ax = axes[row_idx, col_idx]
            plot_df = summary[(summary["keyword"] == keyword) & (summary["party"] == party)]
            plot_df = plot_df.sort_values("year")

            bottom = np.zeros(len(YEARS), dtype=float)
            for stance in STANCES:
                values = plot_df[f"{stance}_pct"].to_numpy(dtype=float)
                ax.bar(
                    x,
                    values,
                    width,
                    bottom=bottom,
                    color=STANCE_COLORS[stance],
                    edgecolor="white",
                    linewidth=1.0,
                )
                for i, value in enumerate(values):
                    annotate_stack(ax, x[i], bottom[i], value, value)
                bottom += values

            totals = plot_df["total"].astype(int).tolist()
            tick_labels = [f"{year}\nn={total:,}" for year, total in zip(YEARS, totals)]
            ax.set_xticks(x)
            ax.set_xticklabels(tick_labels, fontsize=7.4)
            ax.set_ylim(0, 100)
            ax.set_yticks(np.arange(0, 101, 25))
            ax.grid(axis="x", visible=False)
            ax.grid(axis="y", linestyle="--", alpha=0.28, color="#bdc3c7")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if row_idx == 0:
                ax.set_title(PARTY_LABELS[party], fontsize=12, pad=4)
            if col_idx == 0:
                ax.set_ylabel(f"{KEYWORD_LABELS[keyword]}\nTweet-keyword rows (%)", fontsize=8.4)
            else:
                ax.set_ylabel("")

    legend_handles = [
        Patch(facecolor=STANCE_COLORS[stance], edgecolor="white", label=STANCE_LABELS[stance])
        for stance in STANCES
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.986),
        frameon=True,
        fancybox=True,
        fontsize=9.4,
    )

    plt.subplots_adjust(top=0.91, bottom=0.055, left=0.12, right=0.995, hspace=0.26, wspace=0.12)
    plt.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    plt.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summary = load_requested_rows()
    plot_rows(summary)
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
