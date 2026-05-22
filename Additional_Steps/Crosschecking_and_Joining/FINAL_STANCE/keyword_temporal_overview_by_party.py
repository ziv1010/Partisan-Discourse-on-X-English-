#!/usr/bin/env python3
"""
Keyword + Temporal Overview Split by Party
==========================================

Creates a 2x2 figure with keyword-by-year heatmaps:
1. Pro Ruling volume
2. Pro Opposition volume
3. Pro Ruling stance balance
4. Pro Opposition stance balance

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

OUTPUT_PNG = BASE_DIR / "keyword_temporal_overview_by_party.png"
OUTPUT_PDF = BASE_DIR / "keyword_temporal_overview_by_party.pdf"
OUTPUT_SUMMARY_CSV = BASE_DIR / "keyword_temporal_summary_by_party.csv"

VALID_STANCES = ["favor", "neutral", "against"]
VALID_PARTIES = ["pro ruling", "pro opposition"]
PARTY_LABELS = {"pro ruling": "Pro Ruling", "pro opposition": "Pro Opposition"}

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 9,
    }
)


def load_final_df() -> pd.DataFrame:
    print("=" * 70)
    print("LOADING FINAL STANCE DATA FOR PARTY-SPLIT KEYWORD-TEMPORAL VIEW")
    print("=" * 70)

    df = pd.read_csv(
        FINAL_CSV,
        usecols=["row_id", "keyword", "fewshot_label", "_label_norm"],
        dtype={"row_id": "string", "keyword": "string", "fewshot_label": "string", "_label_norm": "string"},
        low_memory=False,
    )
    df["row_id"] = df["row_id"].fillna("").astype(str).str.strip()
    df["keyword"] = df["keyword"].fillna("").astype(str).str.lower().str.strip()
    df["fewshot_label"] = df["fewshot_label"].fillna("").astype(str).str.lower().str.strip()
    df["party"] = df["_label_norm"].fillna("").astype(str).str.lower().str.strip()
    df = df[
        (df["row_id"] != "")
        & (df["keyword"] != "")
        & (df["fewshot_label"].isin(VALID_STANCES))
        & (df["party"].isin(VALID_PARTIES))
    ].copy()

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
        merged.groupby(["keyword", "year", "party"], as_index=False)
        .size()
        .rename(columns={"size": "tweet_keyword_rows"})
    )

    stance_counts = (
        merged.groupby(["keyword", "year", "party", "fewshot_label"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
    )
    stance_pivot = (
        stance_counts.pivot(index=["keyword", "year", "party"], columns="fewshot_label", values="n")
        .fillna(0)
        .reindex(columns=VALID_STANCES, fill_value=0)
        .reset_index()
    )
    stance_pivot["total"] = stance_pivot[VALID_STANCES].sum(axis=1)
    stance_pivot["favor_pct"] = np.where(stance_pivot["total"] > 0, stance_pivot["favor"] / stance_pivot["total"] * 100.0, 0.0)
    stance_pivot["against_pct"] = np.where(stance_pivot["total"] > 0, stance_pivot["against"] / stance_pivot["total"] * 100.0, 0.0)
    stance_pivot["stance_balance"] = stance_pivot["favor_pct"] - stance_pivot["against_pct"]

    summary = counts.merge(
        stance_pivot[
            ["keyword", "year", "party", "favor", "neutral", "against", "total", "favor_pct", "against_pct", "stance_balance"]
        ],
        on=["keyword", "year", "party"],
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


def build_matrix(summary: pd.DataFrame, keyword_order: list[str], years: list[str], party: str, value_col: str) -> pd.DataFrame:
    return (
        summary[summary["party"] == party]
        .pivot(index="keyword", columns="year", values=value_col)
        .reindex(index=keyword_order, columns=years)
        .fillna(0)
    )


def annotate_heatmap(ax, matrix: pd.DataFrame, fmt: str, fontsize: int) -> None:
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            if value != 0:
                ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=fontsize, color="black")


def plot_overview(summary: pd.DataFrame, keyword_order: list[str], years: list[str]) -> None:
    vol_ruling = build_matrix(summary, keyword_order, years, "pro ruling", "tweet_keyword_rows")
    vol_opp = build_matrix(summary, keyword_order, years, "pro opposition", "tweet_keyword_rows")
    bal_ruling = build_matrix(summary, keyword_order, years, "pro ruling", "stance_balance")
    bal_opp = build_matrix(summary, keyword_order, years, "pro opposition", "stance_balance")

    volume_vmax = max(vol_ruling.to_numpy().max(), vol_opp.to_numpy().max(), 1)
    balance_vmax = max(
        np.abs(bal_ruling.to_numpy()).max(),
        np.abs(bal_opp.to_numpy()).max(),
        1.0,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(17, max(14, 0.42 * len(keyword_order))),
        constrained_layout=True,
    )

    panels = [
        (axes[0, 0], vol_ruling, f"{PARTY_LABELS['pro ruling']} Volume", "Blues", 0, volume_vmax, "{:,.0f}", 6),
        (axes[0, 1], vol_opp, f"{PARTY_LABELS['pro opposition']} Volume", "Blues", 0, volume_vmax, "{:,.0f}", 6),
        (axes[1, 0], bal_ruling, f"{PARTY_LABELS['pro ruling']} Stance Balance", "RdYlGn", -balance_vmax, balance_vmax, "{:.0f}", 6),
        (axes[1, 1], bal_opp, f"{PARTY_LABELS['pro opposition']} Stance Balance", "RdYlGn", -balance_vmax, balance_vmax, "{:.0f}", 6),
    ]

    for ax, matrix, title, cmap, vmin, vmax, fmt, fs in panels:
        im = ax.imshow(matrix.to_numpy(), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(years)))
        ax.set_xticklabels(years)
        ax.set_yticks(np.arange(len(keyword_order)))
        ax.set_yticklabels([kw.title() for kw in keyword_order])
        ax.set_xlabel("Year")
        annotate_heatmap(ax, matrix, fmt, fs)

        cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
        cbar.set_label("Tweet-Keyword Rows" if "Volume" in title else "Favor% - Against%")

    axes[0, 0].set_ylabel("Keyword")
    axes[1, 0].set_ylabel("Keyword")

    peak_ruling = summary[summary["party"] == "pro ruling"].loc[summary[summary["party"] == "pro ruling"]["tweet_keyword_rows"].idxmax()]
    peak_opp = summary[summary["party"] == "pro opposition"].loc[summary[summary["party"] == "pro opposition"]["tweet_keyword_rows"].idxmax()]
    shift_ruling = summary[summary["party"] == "pro ruling"].loc[summary[summary["party"] == "pro ruling"]["stance_balance"].abs().idxmax()]
    shift_opp = summary[summary["party"] == "pro opposition"].loc[summary[summary["party"] == "pro opposition"]["stance_balance"].abs().idxmax()]

    fig.suptitle("Keyword and Temporal Overview Split by Party", fontsize=16, y=1.01)
    fig.text(
        0.5,
        0.01,
        (
            f"Peak Pro Ruling volume: {peak_ruling['keyword'].title()} in {peak_ruling['year']} "
            f"({int(peak_ruling['tweet_keyword_rows']):,}). "
            f"Peak Pro Opposition volume: {peak_opp['keyword'].title()} in {peak_opp['year']} "
            f"({int(peak_opp['tweet_keyword_rows']):,}). "
            f"Largest Pro Ruling balance shift: {shift_ruling['keyword'].title()} in {shift_ruling['year']} "
            f"({shift_ruling['stance_balance']:.1f}). "
            f"Largest Pro Opposition balance shift: {shift_opp['keyword'].title()} in {shift_opp['year']} "
            f"({shift_opp['stance_balance']:.1f})."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
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
    print("KEYWORD TEMPORAL OVERVIEW BY PARTY COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
