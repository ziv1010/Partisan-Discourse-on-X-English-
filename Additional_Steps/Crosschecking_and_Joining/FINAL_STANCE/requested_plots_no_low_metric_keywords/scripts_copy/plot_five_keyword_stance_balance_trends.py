#!/usr/bin/env python3
"""
Alternative plot for the five requested keywords.

This figure tracks stance balance over time:
    stance balance = favor percentage - against percentage

Positive values mean the keyword-party-year cell is more favorable; negative
values mean it is more against.

Exact input dataset copy used:
    scripts_copy/five_keyword_stance_by_year_party_summary.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PLOT_DIR = Path(
    "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/"
    "Additional_Steps/Crosschecking_and_Joining/FINAL_STANCE/"
    "requested_plots_no_low_metric_keywords"
)
SCRIPT_DIR = Path(__file__).parent.resolve()
SOURCE_CSV = SCRIPT_DIR / "five_keyword_stance_by_year_party_summary.csv"
OUTPUT_CSV = PLOT_DIR / "five_keyword_stance_balance_trends_summary.csv"
OUTPUT_PNG = PLOT_DIR / "five_keyword_stance_balance_trends.png"
OUTPUT_PDF = PLOT_DIR / "five_keyword_stance_balance_trends.pdf"
INPUT_DATASETS = {
    "five_keyword_stance_by_year_party_summary_csv": SOURCE_CSV,
}

KEYWORDS = ["bhakts", "democracy", "demonetisation", "modi", "suicides"]
YEARS = ["2020", "2021", "2022", "2023"]
PARTIES = ["pro ruling", "pro opposition"]

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

PARTY_COLORS = {
    "pro ruling": "#2166ac",
    "pro opposition": "#b2182b",
}

PARTY_MARKERS = {
    "pro ruling": "o",
    "pro opposition": "s",
}


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 15,
        "axes.labelsize": 15,
        "axes.titlesize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 15,
        "figure.titlesize": 22,
        "savefig.facecolor": "white",
    }
)


def load_summary() -> pd.DataFrame:
    input_csv = SOURCE_CSV
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_csv}")

    df = pd.read_csv(input_csv, dtype={"year": "string"})
    df["keyword"] = df["keyword"].astype(str).str.lower().str.strip()
    df["party"] = df["party"].astype(str).str.lower().str.strip()
    df["year"] = df["year"].astype(str).str.strip()

    summary = df[
        df["keyword"].isin(KEYWORDS)
        & df["party"].isin(PARTIES)
        & df["year"].isin(YEARS)
    ].copy()

    summary["stance_balance"] = summary["favor_pct"] - summary["against_pct"]
    summary["keyword"] = pd.Categorical(summary["keyword"], categories=KEYWORDS, ordered=True)
    summary["party"] = pd.Categorical(summary["party"], categories=PARTIES, ordered=True)
    summary["year"] = pd.Categorical(summary["year"], categories=YEARS, ordered=True)
    summary = summary.sort_values(["keyword", "party", "year"]).reset_index(drop=True)
    summary.to_csv(OUTPUT_CSV, index=False)
    return summary


def plot_balance(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(12.5, 12.8),
        sharex=True,
        sharey=True,
        facecolor="white",
    )
    axes_flat = axes.flatten()
    x = np.arange(len(YEARS))

    for idx, keyword in enumerate(KEYWORDS):
        ax = axes_flat[idx]
        keyword_df = summary[summary["keyword"] == keyword]
        ax.axhline(0, color="#222222", linewidth=1.0, alpha=0.7)

        for party in PARTIES:
            party_df = keyword_df[keyword_df["party"] == party].sort_values("year")
            y = party_df["stance_balance"].to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                color=PARTY_COLORS[party],
                marker=PARTY_MARKERS[party],
                linewidth=2.4,
                markersize=6,
                label=PARTY_LABELS[party],
            )

            for x_pos, y_val in zip(x, y):
                vertical_offset = 4 if y_val >= 0 else -6
                va = "bottom" if y_val >= 0 else "top"
                ax.text(
                    x_pos,
                    y_val + vertical_offset,
                    f"{y_val:.0f}",
                    ha="center",
                    va=va,
                    fontsize=11,
                    color=PARTY_COLORS[party],
                    fontweight="bold",
                )

        ax.set_title(KEYWORD_LABELS[keyword], fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(YEARS)
        ax.set_ylim(-100, 100)
        ax.set_yticks(np.arange(-100, 101, 50))
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Year")

        if idx % 2 == 0:
            ax.set_ylabel("Favor % - Against %")

    axes_flat[-1].axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.958),
        frameon=False,
    )

    plt.subplots_adjust(top=0.94, bottom=0.055, left=0.085, right=0.985, hspace=0.28, wspace=0.04)
    plt.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    plt.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summary = load_summary()
    plot_balance(summary)
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
