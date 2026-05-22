#!/usr/bin/env python3
"""
Keyword + Temporal Overview
===========================

Builds one combined figure with two keyword-by-year heatmaps:
1. Volume heatmap: tweet-keyword rows by year
2. Stance-balance heatmap: (favor% - against%) by year

Unit of analysis:
- One row in the final CSV is one unique (tweet, keyword) pair.
- This is the correct unit for keyword-over-time analysis.
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

OUTPUT_PNG = BASE_DIR / "keyword_temporal_overview.png"
OUTPUT_PDF = BASE_DIR / "keyword_temporal_overview.pdf"
OUTPUT_SUMMARY_CSV = BASE_DIR / "keyword_temporal_summary.csv"

VALID_STANCES = ["favor", "neutral", "against"]

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 10,
    }
)


def load_final_df() -> pd.DataFrame:
    print("=" * 70)
    print("LOADING FINAL STANCE DATA FOR KEYWORD-TEMPORAL VIEW")
    print("=" * 70)

    df = pd.read_csv(
        FINAL_CSV,
        usecols=["row_id", "keyword", "fewshot_label"],
        dtype={"row_id": "string", "keyword": "string", "fewshot_label": "string"},
        low_memory=False,
    )
    df["row_id"] = df["row_id"].fillna("").astype(str).str.strip()
    df["keyword"] = df["keyword"].fillna("").astype(str).str.lower().str.strip()
    df["fewshot_label"] = df["fewshot_label"].fillna("").astype(str).str.lower().str.strip()
    df = df[(df["row_id"] != "") & (df["keyword"] != "") & (df["fewshot_label"].isin(VALID_STANCES))].copy()

    print(f"Rows loaded: {len(df):,}")
    print(f"Unique keywords: {df['keyword'].nunique()}")
    return df


def load_year_lookup(target_row_ids: set[str]) -> pd.DataFrame:
    print("\nMatching years from bridge CSV...")
    chunks = []
    for i, chunk in enumerate(
        pd.read_csv(BRIDGE_CSV, usecols=["row_id", "timestamp"], dtype="string", chunksize=250000, low_memory=False)
    ):
        chunk["row_id"] = chunk["row_id"].fillna("").astype(str).str.strip()
        chunk["timestamp"] = chunk["timestamp"].fillna("").astype(str).str.strip()
        matched = chunk[chunk["row_id"].isin(target_row_ids)].copy()
        if not matched.empty:
            matched["year"] = matched["timestamp"].str.slice(0, 4)
            chunks.append(matched[["row_id", "year"]])
        if (i + 1) % 10 == 0:
            print(f"  Processed {(i + 1) * 250000:,} bridge rows...")

    lookup = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["row_id"], keep="first")
    print(f"Matched row_ids: {len(lookup):,}")
    return lookup


def build_summary(df: pd.DataFrame, year_lookup: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    merged = df.merge(year_lookup, on="row_id", how="left", validate="many_to_one")
    missing_years = merged["year"].isna().sum()
    if missing_years:
        raise ValueError(f"{missing_years:,} rows are missing year after the bridge join.")

    years = sorted(merged["year"].dropna().unique().tolist())

    counts = (
        merged.groupby(["keyword", "year"], as_index=False)
        .size()
        .rename(columns={"size": "tweet_keyword_rows"})
    )

    stance_counts = (
        merged.groupby(["keyword", "year", "fewshot_label"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
    )
    stance_pivot = (
        stance_counts.pivot(index=["keyword", "year"], columns="fewshot_label", values="n")
        .fillna(0)
        .reindex(columns=VALID_STANCES, fill_value=0)
        .reset_index()
    )
    stance_pivot["total"] = stance_pivot[VALID_STANCES].sum(axis=1)
    stance_pivot["favor_pct"] = np.where(stance_pivot["total"] > 0, stance_pivot["favor"] / stance_pivot["total"] * 100.0, 0.0)
    stance_pivot["against_pct"] = np.where(stance_pivot["total"] > 0, stance_pivot["against"] / stance_pivot["total"] * 100.0, 0.0)
    stance_pivot["stance_balance"] = stance_pivot["favor_pct"] - stance_pivot["against_pct"]

    summary = counts.merge(
        stance_pivot[["keyword", "year", "favor", "neutral", "against", "total", "favor_pct", "against_pct", "stance_balance"]],
        on=["keyword", "year"],
        how="left",
        validate="one_to_one",
    )

    keyword_order = (
        summary.groupby("keyword")["tweet_keyword_rows"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )

    return summary, keyword_order, years


def plot_overview(summary: pd.DataFrame, keyword_order: list[str], years: list[str]) -> None:
    counts_mat = (
        summary.pivot(index="keyword", columns="year", values="tweet_keyword_rows")
        .reindex(index=keyword_order, columns=years)
        .fillna(0)
    )
    balance_mat = (
        summary.pivot(index="keyword", columns="year", values="stance_balance")
        .reindex(index=keyword_order, columns=years)
        .fillna(0)
    )

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(15, max(12, 0.38 * len(keyword_order))),
        gridspec_kw={"width_ratios": [1, 1]},
        constrained_layout=True,
    )

    im1 = ax1.imshow(counts_mat.to_numpy(), aspect="auto", cmap="Blues")
    ax1.set_title("Keyword Volume by Year\n(Tweet-Keyword Rows)")
    ax1.set_xticks(np.arange(len(years)))
    ax1.set_xticklabels(years)
    ax1.set_yticks(np.arange(len(keyword_order)))
    ax1.set_yticklabels([kw.title() for kw in keyword_order])
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Keyword")

    vmax = np.nanmax(np.abs(balance_mat.to_numpy()))
    vmax = max(vmax, 1.0)
    im2 = ax2.imshow(balance_mat.to_numpy(), aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax2.set_title("Keyword Stance Balance by Year\n(Favor% - Against%)")
    ax2.set_xticks(np.arange(len(years)))
    ax2.set_xticklabels(years)
    ax2.set_yticks(np.arange(len(keyword_order)))
    ax2.set_yticklabels([kw.title() for kw in keyword_order])
    ax2.set_xlabel("Year")

    for i in range(len(keyword_order)):
        for j in range(len(years)):
            count_val = int(counts_mat.iloc[i, j])
            if count_val > 0:
                ax1.text(j, i, f"{count_val:,}", ha="center", va="center", fontsize=7, color="#0f172a")

            bal = balance_mat.iloc[i, j]
            if counts_mat.iloc[i, j] > 0:
                ax2.text(
                    j,
                    i,
                    f"{bal:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )

    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03)
    cbar1.set_label("Tweet-Keyword Rows")
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03)
    cbar2.set_label("Favor% - Against%")

    top_keyword = keyword_order[0]
    peak_year_row = summary.loc[summary["tweet_keyword_rows"].idxmax()]
    balance_peak = summary.loc[summary["stance_balance"].abs().idxmax()]

    fig.suptitle(
        "Keyword and Temporal Overview of Final Stance Dataset",
        fontsize=16,
        y=1.01,
    )
    fig.text(
        0.5,
        0.01,
        (
            f"Rows are sorted by total keyword volume. Most frequent keyword: {top_keyword.title()}. "
            f"Peak keyword-year volume: {peak_year_row['keyword'].title()} in {peak_year_row['year']} "
            f"({int(peak_year_row['tweet_keyword_rows']):,} rows). "
            f"Largest stance balance shift: {balance_peak['keyword'].title()} in {balance_peak['year']} "
            f"({balance_peak['stance_balance']:.1f} points)."
        ),
        ha="center",
        va="bottom",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="#d0d0d0", boxstyle="round,pad=0.35"),
    )

    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight", format="pdf")
    plt.close()

    print(f"\nSaved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")


def main() -> None:
    df = load_final_df()
    year_lookup = load_year_lookup(set(df["row_id"].unique()))
    summary, keyword_order, years = build_summary(df, year_lookup)
    summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)

    print(f"\nKeywords in plot: {len(keyword_order)}")
    print(f"Years in plot: {years}")
    print(f"Saved: {OUTPUT_SUMMARY_CSV}")

    plot_overview(summary, keyword_order, years)

    print("\n" + "=" * 70)
    print("KEYWORD TEMPORAL OVERVIEW COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
