#!/usr/bin/env python3
"""
Yearly Temporal Analysis for the Final Stance Dataset
====================================================

Generates:
1. Unique tweet volume by year
2. Yearly stance composition using stance-labeled tweet-keyword rows
3. A summary CSV with yearly counts and percentages

Why two different units?
- Volume plot uses unique tweets (`row_id`) because the same tweet can match
  multiple keywords.
- Stance-over-time plot uses tweet-keyword rows because stance is assigned at
  that level, and many tweets have different stance labels for different
  keywords.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).parent.resolve()
FINAL_CSV = BASE_DIR / "extracted_ALL_35keywords_english_hindi_only_final_with_existing_and_inferred_stance.csv"
BRIDGE_CSV = Path(
    "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Additional_Steps/Crosschecking_and_Joining/SCRIPTS_OuTPUTS/retweet_data_combined_labeled_by_retweet_en_with_hinglish_bridge.csv"
)

OUTPUT_COUNTS_PNG = BASE_DIR / "temporal_unique_tweets_by_year.png"
OUTPUT_COUNTS_PDF = BASE_DIR / "temporal_unique_tweets_by_year.pdf"
OUTPUT_STANCE_PNG = BASE_DIR / "temporal_stance_over_years.png"
OUTPUT_STANCE_PDF = BASE_DIR / "temporal_stance_over_years.pdf"
OUTPUT_SUMMARY_CSV = BASE_DIR / "temporal_summary_by_year.csv"

VALID_STANCES = ["favor", "neutral", "against"]
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
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
)


def load_final_rows() -> pd.DataFrame:
    print("=" * 70)
    print("LOADING FINAL STANCE CSV")
    print("=" * 70)
    print(f"Final CSV: {FINAL_CSV}")

    df = pd.read_csv(
        FINAL_CSV,
        usecols=["row_id", "fewshot_label"],
        dtype={"row_id": "string", "fewshot_label": "string"},
        low_memory=False,
    )
    df["row_id"] = df["row_id"].fillna("").astype(str).str.strip()
    df["fewshot_label"] = df["fewshot_label"].fillna("").astype(str).str.lower().str.strip()
    df = df[df["row_id"] != ""].copy()
    df = df[df["fewshot_label"].isin(VALID_STANCES)].copy()

    print(f"Rows with valid stance labels: {len(df):,}")
    print(f"Unique tweet ids: {df['row_id'].nunique():,}")
    return df


def build_year_lookup(target_row_ids: set[str]) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("MATCHING TIMESTAMPS FROM BRIDGE CSV")
    print("=" * 70)
    print(f"Bridge CSV: {BRIDGE_CSV}")

    chunks = []
    total_matched = 0
    for i, chunk in enumerate(
        pd.read_csv(BRIDGE_CSV, usecols=["row_id", "timestamp"], dtype="string", chunksize=250000, low_memory=False)
    ):
        chunk["row_id"] = chunk["row_id"].fillna("").astype(str).str.strip()
        chunk["timestamp"] = chunk["timestamp"].fillna("").astype(str).str.strip()
        matched = chunk[chunk["row_id"].isin(target_row_ids)].copy()
        if not matched.empty:
            matched["year"] = matched["timestamp"].str.slice(0, 4)
            chunks.append(matched[["row_id", "year"]])
            total_matched += len(matched)
        if (i + 1) % 10 == 0:
            print(f"  Processed {(i + 1) * 250000:,} bridge rows...")

    lookup = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["row_id"], keep="first")
    print(f"Matched row_ids with timestamps: {len(lookup):,}")
    return lookup


def build_yearly_summary(df_rows: pd.DataFrame, year_lookup: pd.DataFrame) -> pd.DataFrame:
    merged = df_rows.merge(year_lookup, on="row_id", how="left", validate="many_to_one")
    missing_years = merged["year"].isna().sum()
    if missing_years:
        raise ValueError(f"{missing_years:,} stance rows are missing year after the bridge join.")

    unique_tweets = (
        merged[["row_id", "year"]]
        .drop_duplicates()
        .groupby("year", as_index=False)
        .size()
        .rename(columns={"size": "unique_tweets"})
    )

    stance_counts = (
        merged.groupby(["year", "fewshot_label"], as_index=False)
        .size()
        .rename(columns={"size": "stance_rows"})
    )
    stance_pivot = (
        stance_counts.pivot(index="year", columns="fewshot_label", values="stance_rows")
        .fillna(0)
        .reindex(columns=VALID_STANCES, fill_value=0)
        .astype(int)
        .reset_index()
    )
    stance_pivot["stance_rows_total"] = stance_pivot[VALID_STANCES].sum(axis=1)
    for stance in VALID_STANCES:
        stance_pivot[f"{stance}_pct"] = np.where(
            stance_pivot["stance_rows_total"] > 0,
            stance_pivot[stance] / stance_pivot["stance_rows_total"] * 100.0,
            0.0,
        )

    summary = unique_tweets.merge(stance_pivot, on="year", how="inner").sort_values("year").reset_index(drop=True)
    return summary


def plot_unique_tweets_by_year(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))

    years = summary["year"].tolist()
    counts = summary["unique_tweets"].to_numpy()
    bars = ax.bar(years, counts, color="#2b6cb0", edgecolor="#16324f", linewidth=1.0, width=0.68)

    peak_idx = int(np.argmax(counts))
    bars[peak_idx].set_color("#d97706")
    bars[peak_idx].set_edgecolor("#92400e")

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.015,
            f"{int(count):,}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.set_xlabel("Year")
    ax.set_ylabel("Unique Tweets")
    ax.set_title("Unique Tweets Per Year")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    peak_year = years[peak_idx]
    ax.text(
        0.98,
        0.96,
        f"Peak year: {peak_year}\n{int(counts[peak_idx]):,} unique tweets",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox=dict(facecolor="white", edgecolor="#d0d0d0", boxstyle="round,pad=0.4"),
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_COUNTS_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_COUNTS_PDF, dpi=300, bbox_inches="tight", format="pdf")
    plt.close()

    print(f"\nSaved: {OUTPUT_COUNTS_PNG}")
    print(f"Saved: {OUTPUT_COUNTS_PDF}")


def plot_stance_over_years(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 7.5))

    years = summary["year"].tolist()
    x = np.arange(len(years))
    bottom = np.zeros(len(years), dtype=float)

    for stance in VALID_STANCES:
        values = summary[f"{stance}_pct"].to_numpy()
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=STANCE_COLORS[stance],
            edgecolor="white",
            linewidth=1.0,
            width=0.72,
            label=stance.capitalize(),
        )

        for i, value in enumerate(values):
            if value >= 11:
                ax.text(
                    x[i],
                    bottom[i] + value / 2,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="black",
                )
        bottom += values

    ax2 = ax.twinx()
    totals = summary["stance_rows_total"].to_numpy()
    ax2.plot(x, totals, color="#1f2937", marker="o", linewidth=2.2, label="Total stance rows")
    for i, total in enumerate(totals):
        ax2.text(x[i], total + totals.max() * 0.02, f"{int(total):,}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Year")
    ax.set_ylabel("Stance Share of Tweet-Keyword Rows (%)")
    ax2.set_ylabel("Total Tweet-Keyword Rows")
    ax.set_title("Stance Composition Over Years")

    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper left",
        frameon=True,
        fancybox=True,
    )

    dominant = summary.loc[summary["favor_pct"].idxmax()]
    ax.text(
        0.98,
        0.96,
        "Note: Stance is plotted on tweet-keyword rows,\nnot deduplicated tweets.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="#d0d0d0", boxstyle="round,pad=0.4"),
    )
    ax.text(
        0.98,
        0.76,
        f"Highest favor share: {dominant['year']}\n{dominant['favor_pct']:.1f}% favor",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="#f7fafc", edgecolor="#d0d0d0", boxstyle="round,pad=0.4"),
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_STANCE_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_STANCE_PDF, dpi=300, bbox_inches="tight", format="pdf")
    plt.close()

    print(f"\nSaved: {OUTPUT_STANCE_PNG}")
    print(f"Saved: {OUTPUT_STANCE_PDF}")


def main() -> None:
    df_rows = load_final_rows()
    year_lookup = build_year_lookup(set(df_rows["row_id"].unique()))
    summary = build_yearly_summary(df_rows, year_lookup)
    summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)

    print("\nYearly summary")
    print(summary.to_string(index=False))
    print(f"\nSaved: {OUTPUT_SUMMARY_CSV}")

    plot_unique_tweets_by_year(summary)
    plot_stance_over_years(summary)

    print("\n" + "=" * 70)
    print("TEMPORAL ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
