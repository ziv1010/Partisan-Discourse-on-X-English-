#!/usr/bin/env python3
"""
Requested final-stance plots with low-metric keywords removed.

Outputs are written beside this script so the regenerated plots are isolated
from the older FINAL_STANCE figures.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")


OUTPUT_DIR = Path(__file__).parent.resolve()
BASE_DIR = OUTPUT_DIR.parent
FINAL_CSV = BASE_DIR / "extracted_ALL_35keywords_english_hindi_only_final_with_existing_and_inferred_stance.csv"
BRIDGE_CSV = (
    BASE_DIR.parent
    / "SCRIPTS_OuTPUTS"
    / "retweet_data_combined_labeled_by_retweet_en_with_hinglish_bridge.csv"
)

VALID_STANCES = ["favor", "neutral", "against"]
VALID_PARTIES = ["pro ruling", "pro opposition"]
LANGUAGES = ["english", "hindi"]
MIN_UNIQUE_PER_PARTY = 2

EXCLUDED_KEYWORDS = {
    "dictatorship",
    "hathras",
    "hindu",
    "hindutva",
    "islamists",
    "kashmir",
    "kashmiri pandits",
    "lynching",
    "balochistan",
    "minorities",
    "msp",
    "sangh",
    "sharia",
    "unemployment",
}

STACKED_KEYWORDS = [
    "hindutva",
    "bhakts",
    "modi",
    "ram mandir",
    "democracy",
    "demonetisation",
    "suicides",
    "rahulgandhi",
    "farmers protests",
    "muslim",
]

STANCE_COLORS = {
    "favor": "#27ae60",
    "neutral": "#95a5a6",
    "against": "#c0392b",
}

LANG_MARKERS = {
    "english": "o",
    "hindi": "s",
}

LANG_LABELS = {
    "english": "English",
    "hindi": "Hindi",
}

PARTY_LABELS = {
    "pro ruling": "Pro Ruling",
    "pro opposition": "Pro Opposition",
}

DISPLAY_OVERRIDES = {
    "aatmanirbhar": "Aatmanirbhar",
    "caa": "CAA",
    "gdp": "GDP",
    "msp": "MSP",
    "rahulgandhi": "Rahul Gandhi",
}

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 10,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 8,
        "figure.titlesize": 11,
        "savefig.facecolor": "white",
    }
)


def display_keyword(keyword: str) -> str:
    keyword = str(keyword).strip().lower()
    return DISPLAY_OVERRIDES.get(keyword, keyword.title())


def derive_language_group(chunk: pd.DataFrame) -> pd.Series:
    language = chunk["language"].fillna("").astype(str).str.lower().str.strip()
    translation_route = chunk["translation_route"].fillna("").astype(str).str.lower().str.strip()
    lid_lang = chunk["lid_lang"].fillna("").astype(str).str.lower().str.strip()

    group = language.copy()
    hinglish_mask = (translation_route == "hinglish_roman") | (lid_lang == "hin_latn")
    group.loc[hinglish_mask] = "hinglish"
    return group


def load_final_rows() -> pd.DataFrame:
    if not FINAL_CSV.exists():
        raise FileNotFoundError(f"Missing final CSV: {FINAL_CSV}")

    print("=" * 72)
    print("LOADING FINAL STANCE ROWS")
    print("=" * 72)
    print(f"Final CSV: {FINAL_CSV}")
    print("Excluded keywords:", ", ".join(display_keyword(k) for k in sorted(EXCLUDED_KEYWORDS)))

    usecols = [
        "source_row",
        "row_id",
        "keyword",
        "fewshot_label",
        "_label_norm",
        "language",
        "translation_route",
        "lid_lang",
    ]
    chunks: list[pd.DataFrame] = []
    total_rows = 0
    kept_rows = 0

    for i, chunk in enumerate(
        pd.read_csv(FINAL_CSV, usecols=usecols, dtype="string", chunksize=250000, low_memory=False)
    ):
        total_rows += len(chunk)
        chunk["row_id"] = chunk["row_id"].fillna("").astype(str).str.strip()
        chunk["source_row"] = chunk["source_row"].fillna("").astype(str).str.strip()
        chunk["keyword"] = chunk["keyword"].fillna("").astype(str).str.lower().str.strip()
        chunk["stance"] = chunk["fewshot_label"].fillna("").astype(str).str.lower().str.strip()
        chunk["party"] = chunk["_label_norm"].fillna("").astype(str).str.lower().str.strip()
        chunk["language_group"] = derive_language_group(chunk)

        chunk = chunk[
            (chunk["row_id"] != "")
            & (chunk["source_row"] != "")
            & (chunk["keyword"] != "")
            & (chunk["stance"].isin(VALID_STANCES))
            & (~chunk["keyword"].isin(EXCLUDED_KEYWORDS))
        ].copy()

        if not chunk.empty:
            chunks.append(chunk[["source_row", "row_id", "keyword", "stance", "party", "language_group"]])
            kept_rows += len(chunk)

        if (i + 1) % 5 == 0:
            print(f"  Processed {total_rows:,} rows; kept {kept_rows:,} after filters...")

    if not chunks:
        raise ValueError("No rows remained after filtering.")

    df = pd.concat(chunks, ignore_index=True)
    print(f"Rows read: {total_rows:,}")
    print(f"Rows kept after stance/keyword filters: {len(df):,}")
    print(f"Keywords kept: {df['keyword'].nunique()}")
    print("Language groups:")
    print(df["language_group"].value_counts().to_string())
    print("Parties:")
    print(df["party"].value_counts().to_string())
    return df


def compute_language_scatter_points(df: pd.DataFrame, language: str) -> pd.DataFrame:
    rows = []
    lang_df = df[(df["language_group"] == language) & (df["party"].isin(VALID_PARTIES))].copy()

    for keyword in sorted(lang_df["keyword"].unique()):
        keyword_df = lang_df[lang_df["keyword"] == keyword]
        ruling = keyword_df[keyword_df["party"] == "pro ruling"]
        opposition = keyword_df[keyword_df["party"] == "pro opposition"]
        ruling_n = ruling["source_row"].nunique()
        opposition_n = opposition["source_row"].nunique()

        if ruling_n < MIN_UNIQUE_PER_PARTY or opposition_n < MIN_UNIQUE_PER_PARTY:
            continue

        ruling_pcts = ruling["stance"].value_counts(normalize=True) * 100.0
        opposition_pcts = opposition["stance"].value_counts(normalize=True) * 100.0
        rows.append(
            {
                "keyword": keyword,
                "language_group": language,
                "favor_diff": ruling_pcts.get("favor", 0.0) - opposition_pcts.get("favor", 0.0),
                "against_diff": ruling_pcts.get("against", 0.0) - opposition_pcts.get("against", 0.0),
                "ruling_unique_tweets": ruling_n,
                "opposition_unique_tweets": opposition_n,
                "total_unique_tweets": ruling_n + opposition_n,
            }
        )

    return pd.DataFrame(rows).sort_values("keyword").reset_index(drop=True)


def plot_language_scatter(points: pd.DataFrame, language: str) -> None:
    if points.empty:
        raise ValueError(f"No qualifying points for {language}.")

    keyword_numbers = {keyword: i + 1 for i, keyword in enumerate(points["keyword"].tolist())}
    x_vals = points["favor_diff"].to_numpy()
    y_vals = points["against_diff"].to_numpy()
    sizes = np.sqrt(points["total_unique_tweets"].to_numpy()) * 120
    color_values = points["favor_diff"].to_numpy()

    fig, ax = plt.subplots(figsize=(17, 12.5))
    ax.axhline(0, color="black", linestyle="-", alpha=0.25, zorder=1)
    ax.axvline(0, color="black", linestyle="-", alpha=0.25, zorder=1)

    scatter = ax.scatter(
        x_vals,
        y_vals,
        s=sizes,
        c=color_values,
        cmap="RdYlGn",
        marker=LANG_MARKERS[language],
        edgecolors="k",
        linewidth=0.6,
        alpha=0.45,
        zorder=2,
    )

    ax.set_xlim(x_vals.min() - 25, x_vals.max() + 25)
    ax.set_ylim(y_vals.min() - 25, y_vals.max() + 25)

    for _, row in points.iterrows():
        txt = ax.text(
            row["favor_diff"],
            row["against_diff"],
            str(keyword_numbers[row["keyword"]]),
            ha="center",
            va="center",
            fontsize=16,
            zorder=4,
            bbox=dict(facecolor="white", edgecolor="#777777", boxstyle="circle,pad=0.18", alpha=0.92),
        )
        txt.set_path_effects([pe.withStroke(linewidth=2.1, foreground="white")])

    ax.set_xlabel("Favor Difference (Ruling % - Opposition %)", fontsize=24)
    ax.set_ylabel("Against Difference (Ruling % - Opposition %)", fontsize=24)
    ax.tick_params(axis="both", labelsize=20)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.036, pad=0.02)
    cbar.set_label("Favor Divergence Score", fontsize=22)
    cbar.ax.tick_params(labelsize=18)

    legend_keywords = "\n".join(
        f"{keyword_numbers[k]}: {display_keyword(k)}" for k in points["keyword"].tolist()
    )
    fig.text(
        0.875,
        0.93,
        "Keyword IDs\n" + legend_keywords,
        ha="left",
        va="top",
        fontsize=15,
        linespacing=1.0,
        bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.45", alpha=0.94),
        family="serif",
    )

    plt.subplots_adjust(left=0.09, right=0.77, top=0.965, bottom=0.105)
    png = OUTPUT_DIR / f"divergence_scatter_{language}_only.png"
    pdf = OUTPUT_DIR / f"divergence_scatter_{language}_only.pdf"
    plt.savefig(png, dpi=600, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight", format="pdf")
    plt.close()

    points.to_csv(OUTPUT_DIR / f"divergence_scatter_{language}_only_points.csv", index=False)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def create_language_scatter_plots(df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("CREATING LANGUAGE-SPECIFIC DIVERGENCE SCATTERS")
    print("=" * 72)
    for language in LANGUAGES:
        points = compute_language_scatter_points(df, language)
        print(f"{LANG_LABELS[language]} qualifying keywords: {len(points)}")
        plot_language_scatter(points, language)


def create_stacked_stance_plot(df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("CREATING ENGLISH/HINDI STACKED STANCE PLOT")
    print("=" * 72)

    plot_df = df[
        (df["party"].isin(VALID_PARTIES))
        & (df["language_group"].isin(LANGUAGES))
        & (df["keyword"].isin([kw for kw in STACKED_KEYWORDS if kw not in EXCLUDED_KEYWORDS]))
    ].copy()
    keywords = [kw for kw in STACKED_KEYWORDS if kw not in EXCLUDED_KEYWORDS and kw in set(plot_df["keyword"])]
    if not keywords:
        raise ValueError("No stacked-plot keywords remained after filtering.")

    bar_groups = [
        ("pro ruling", "english", "Ruling\nEnglish"),
        ("pro ruling", "hindi", "Ruling\nHindi"),
        ("pro opposition", "english", "Opp.\nEnglish"),
        ("pro opposition", "hindi", "Opp.\nHindi"),
    ]

    n_cols = 3
    n_rows = math.ceil(len(keywords) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.2, 6.6), facecolor="white")
    axes = np.atleast_1d(axes).flatten()

    summary_rows = []
    for i, keyword in enumerate(keywords):
        ax = axes[i]
        keyword_df = plot_df[plot_df["keyword"] == keyword]
        bar_data = []

        for party, language, label in bar_groups:
            subset = keyword_df[
                (keyword_df["party"] == party) & (keyword_df["language_group"] == language)
            ]
            n = len(subset)
            counts = subset["stance"].value_counts()
            row = {
                "keyword": keyword,
                "party": party,
                "language_group": language,
                "label": label,
                "n": n,
            }
            for stance in VALID_STANCES:
                count = int(counts.get(stance, 0))
                pct = count / n * 100.0 if n else 0.0
                row[stance] = pct
                row[f"{stance}_n"] = count
            bar_data.append(row)
            summary_rows.append(row.copy())

        x_positions = np.arange(len(bar_data))
        bottom = np.zeros(len(bar_data), dtype=float)
        bar_width = 0.62

        for stance in VALID_STANCES:
            heights = np.array([d[stance] for d in bar_data])
            ax.bar(
                x_positions,
                heights,
                bar_width,
                bottom=bottom,
                color=STANCE_COLORS[stance],
                edgecolor="white",
                linewidth=1.0,
                label=stance.capitalize(),
            )
            for j, (height, base, item) in enumerate(zip(heights, bottom, bar_data)):
                if height >= 10:
                    ax.text(
                        x_positions[j],
                        base + height / 2,
                        f"{height:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=5.7,
                        color="black",
                        linespacing=1.0,
                    )
            bottom += heights

        ax.set_title(display_keyword(keyword), fontsize=10.8, pad=3)
        ax.set_ylim(0, 100)
        ax.set_xlim(-0.5, len(bar_data) - 0.5)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([d["label"] for d in bar_data], fontsize=5.8)
        if i % n_cols == 0:
            ax.set_ylabel("Tweet-keyword rows (%)", fontsize=7.0)
        else:
            ax.set_ylabel("")
        ax.grid(axis="y", linestyle="--", alpha=0.28, color="#bdc3c7")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[len(keywords) :]:
        ax.set_axis_off()

    legend = [
        Patch(facecolor=STANCE_COLORS["favor"], edgecolor="white", label="Favor"),
        Patch(facecolor=STANCE_COLORS["neutral"], edgecolor="white", label="Neutral"),
        Patch(facecolor=STANCE_COLORS["against"], edgecolor="white", label="Against"),
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.986),
        ncol=3,
        frameon=True,
        fancybox=True,
        fontsize=8.4,
    )
    plt.subplots_adjust(top=0.885, bottom=0.055, left=0.075, right=0.995, hspace=0.30, wspace=0.12)

    png = OUTPUT_DIR / "stance_stacked_english_hindi_only.png"
    pdf = OUTPUT_DIR / "stance_stacked_english_hindi_only.pdf"
    plt.savefig(png, dpi=600, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight", format="pdf")
    plt.close()

    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "stance_stacked_english_hindi_only_summary.csv", index=False)
    print(f"Keywords plotted: {len(keywords)}")
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def load_year_lookup(target_row_ids: set[str]) -> pd.DataFrame:
    if not BRIDGE_CSV.exists():
        raise FileNotFoundError(f"Missing bridge CSV: {BRIDGE_CSV}")

    print("\n" + "=" * 72)
    print("MATCHING YEARS FROM BRIDGE CSV")
    print("=" * 72)
    print(f"Bridge CSV: {BRIDGE_CSV}")
    chunks: list[pd.DataFrame] = []
    for i, chunk in enumerate(
        pd.read_csv(BRIDGE_CSV, usecols=["row_id", "timestamp"], dtype="string", chunksize=250000, low_memory=False)
    ):
        chunk["row_id"] = chunk["row_id"].fillna("").astype(str).str.strip()
        chunk["timestamp"] = chunk["timestamp"].fillna("").astype(str).str.strip()
        matched = chunk[chunk["row_id"].isin(target_row_ids)].copy()
        if not matched.empty:
            matched["year"] = matched["timestamp"].str.slice(0, 4)
            matched = matched[matched["year"].str.fullmatch(r"\d{4}", na=False)]
            chunks.append(matched[["row_id", "year"]])
        if (i + 1) % 10 == 0:
            print(f"  Processed {(i + 1) * 250000:,} bridge rows...")

    if not chunks:
        raise ValueError("No matching row_ids were found in the bridge CSV.")

    lookup = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["row_id"], keep="first")
    print(f"Matched row_ids with years: {len(lookup):,}")
    return lookup


def build_yearly_summary(df: pd.DataFrame, year_lookup: pd.DataFrame) -> pd.DataFrame:
    merged = df[["row_id", "stance"]].merge(year_lookup, on="row_id", how="left", validate="many_to_one")
    missing_years = int(merged["year"].isna().sum())
    if missing_years:
        raise ValueError(f"{missing_years:,} rows are missing year after bridge join.")

    unique_tweets = (
        merged[["row_id", "year"]]
        .drop_duplicates()
        .groupby("year", as_index=False)
        .size()
        .rename(columns={"size": "unique_tweets"})
    )
    stance_counts = (
        merged.groupby(["year", "stance"], as_index=False)
        .size()
        .rename(columns={"size": "stance_rows"})
    )
    stance_pivot = (
        stance_counts.pivot(index="year", columns="stance", values="stance_rows")
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

    return unique_tweets.merge(stance_pivot, on="year", how="inner").sort_values("year").reset_index(drop=True)


def plot_temporal_stance_without_total_line(summary: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("CREATING TEMPORAL STANCE PLOT WITHOUT TOTAL LINE")
    print("=" * 72)

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
                    fontsize=15,
                    color="black",
                )
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Year")
    ax.set_ylabel("Stance Share of Tweet-Keyword Rows (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", frameon=True, fancybox=True)

    plt.tight_layout(pad=0.45)
    png = OUTPUT_DIR / "temporal_stance_over_years_no_total_line.png"
    pdf = OUTPUT_DIR / "temporal_stance_over_years_no_total_line.pdf"
    plt.savefig(png, dpi=600, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight", format="pdf")
    plt.close()

    summary.to_csv(OUTPUT_DIR / "temporal_summary_by_year_no_low_metric_keywords.csv", index=False)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def build_keyword_party_summary(df: pd.DataFrame, year_lookup: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    party_df = df[df["party"].isin(VALID_PARTIES)][["row_id", "keyword", "party", "stance"]].copy()
    merged = party_df.merge(year_lookup, on="row_id", how="left", validate="many_to_one")
    missing_years = int(merged["year"].isna().sum())
    if missing_years:
        raise ValueError(f"{missing_years:,} party rows are missing year after bridge join.")

    years = sorted(merged["year"].dropna().unique().tolist())
    counts = (
        merged.groupby(["keyword", "year", "party", "stance"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
    )
    pivot = (
        counts.pivot(index=["keyword", "year", "party"], columns="stance", values="n")
        .fillna(0)
        .reindex(columns=VALID_STANCES, fill_value=0)
        .reset_index()
    )
    pivot[VALID_STANCES] = pivot[VALID_STANCES].astype(int)
    pivot["total"] = pivot[VALID_STANCES].sum(axis=1)
    for stance in VALID_STANCES:
        pivot[f"{stance}_pct"] = np.where(pivot["total"] > 0, pivot[stance] / pivot["total"] * 100.0, 0.0)

    keyword_order = (
        pivot.groupby("keyword")["total"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    return pivot, keyword_order, years


def build_pct_matrix(
    summary: pd.DataFrame,
    keyword_order: list[str],
    years: list[str],
    party: str,
    stance: str,
) -> pd.DataFrame:
    return (
        summary[summary["party"] == party]
        .pivot(index="keyword", columns="year", values=f"{stance}_pct")
        .reindex(index=keyword_order, columns=years)
        .fillna(0.0)
    )


def build_total_matrix(
    summary: pd.DataFrame,
    keyword_order: list[str],
    years: list[str],
    party: str,
) -> pd.DataFrame:
    return (
        summary[summary["party"] == party]
        .pivot(index="keyword", columns="year", values="total")
        .reindex(index=keyword_order, columns=years)
        .fillna(0)
    )


def plot_keyword_party_stance_heatmaps(
    summary: pd.DataFrame,
    keyword_order: list[str],
    years: list[str],
) -> None:
    print("\n" + "=" * 72)
    print("CREATING KEYWORD/PARTY/YEAR STANCE HEATMAPS")
    print("=" * 72)

    stance_cmaps = {
        "favor": "Greens",
        "neutral": "Greys",
        "against": "Reds",
    }
    fig_height = max(13, 0.43 * len(keyword_order))
    fig, axes = plt.subplots(
        len(VALID_PARTIES),
        len(VALID_STANCES),
        figsize=(21, fig_height),
        constrained_layout=True,
        facecolor="white",
    )
    fig.set_constrained_layout_pads(w_pad=0.035, h_pad=0.06, hspace=0.075, wspace=0.04)

    for party_idx, party in enumerate(VALID_PARTIES):
        totals = build_total_matrix(summary, keyword_order, years, party)
        for stance_idx, stance in enumerate(VALID_STANCES):
            ax = axes[party_idx, stance_idx]
            matrix = build_pct_matrix(summary, keyword_order, years, party, stance)
            im = ax.imshow(matrix.to_numpy(), aspect="auto", cmap=stance_cmaps[stance], vmin=0, vmax=100)
            ax.set_title(f"{PARTY_LABELS[party]}: {stance.capitalize()} %")
            ax.set_xticks(np.arange(len(years)))
            ax.set_xticklabels(years)
            ax.set_yticks(np.arange(len(keyword_order)))
            if stance_idx == 0:
                ax.set_yticklabels([display_keyword(k) for k in keyword_order])
                ax.set_ylabel("Keyword")
            else:
                ax.set_yticklabels([])
            ax.set_xlabel("Year")

            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    total = int(totals.iloc[i, j])
                    if total > 0:
                        value = matrix.iloc[i, j]
                        ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=8.5, color="black")

            cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
            cbar.set_label("Share (%)")

    png = OUTPUT_DIR / "keyword_party_stance_percent_by_year.png"
    pdf = OUTPUT_DIR / "keyword_party_stance_percent_by_year.pdf"
    plt.savefig(png, dpi=600, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight", format="pdf")
    plt.close()

    summary.to_csv(OUTPUT_DIR / "keyword_party_stance_percent_by_year_summary.csv", index=False)
    print(f"Keywords plotted: {len(keyword_order)}")
    print(f"Years plotted: {', '.join(years)}")
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def plot_keyword_party_stance_heatmaps_split(
    summary: pd.DataFrame,
    keyword_order: list[str],
    years: list[str],
) -> None:
    """Create one larger heatmap figure per party for paper readability."""
    print("\n" + "=" * 72)
    print("CREATING SPLIT PAPER-READY KEYWORD/PARTY HEATMAPS")
    print("=" * 72)

    stance_cmaps = {
        "favor": "Greens",
        "neutral": "Greys",
        "against": "Reds",
    }

    for party in VALID_PARTIES:
        fig_height = max(11, 0.52 * len(keyword_order))
        fig, axes = plt.subplots(
            1,
            len(VALID_STANCES),
            figsize=(21, fig_height),
            constrained_layout=True,
            facecolor="white",
        )
        fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.04, hspace=0.04, wspace=0.035)
        totals = build_total_matrix(summary, keyword_order, years, party)

        for stance_idx, stance in enumerate(VALID_STANCES):
            ax = axes[stance_idx]
            matrix = build_pct_matrix(summary, keyword_order, years, party, stance)
            im = ax.imshow(matrix.to_numpy(), aspect="auto", cmap=stance_cmaps[stance], vmin=0, vmax=100)
            ax.set_title(f"{stance.capitalize()} %", pad=10, fontsize=24)
            ax.set_xticks(np.arange(len(years)))
            ax.set_xticklabels(years, fontsize=18)
            ax.set_yticks(np.arange(len(keyword_order)))
            if stance_idx == 0:
                ax.set_yticklabels([display_keyword(k) for k in keyword_order], fontsize=17)
                ax.set_ylabel("Keyword", fontsize=20)
            else:
                ax.set_yticklabels([])
            ax.set_xlabel("Year", fontsize=20)

            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    total = int(totals.iloc[i, j])
                    if total > 0:
                        value = matrix.iloc[i, j]
                        ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=10.5, color="black")

            cbar = fig.colorbar(im, ax=ax, fraction=0.034, pad=0.02)
            cbar.set_label("Share (%)", fontsize=17)
            cbar.ax.tick_params(labelsize=15)

        party_slug = party.replace(" ", "_")
        png = OUTPUT_DIR / f"keyword_party_stance_percent_by_year_{party_slug}.png"
        pdf = OUTPUT_DIR / f"keyword_party_stance_percent_by_year_{party_slug}.pdf"
        plt.savefig(png, dpi=600, bbox_inches="tight")
        plt.savefig(pdf, bbox_inches="tight", format="pdf")
        plt.close()
        print(f"Saved: {png}")
        print(f"Saved: {pdf}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_final_rows()

    create_language_scatter_plots(df)
    create_stacked_stance_plot(df)

    year_lookup = load_year_lookup(set(df["row_id"].unique()))
    yearly_summary = build_yearly_summary(df, year_lookup)
    print("\nYearly stance summary after keyword exclusions:")
    print(yearly_summary.to_string(index=False))
    plot_temporal_stance_without_total_line(yearly_summary)

    keyword_party_summary, keyword_order, years = build_keyword_party_summary(df, year_lookup)
    plot_keyword_party_stance_heatmaps(keyword_party_summary, keyword_order, years)
    plot_keyword_party_stance_heatmaps_split(keyword_party_summary, keyword_order, years)

    print("\n" + "=" * 72)
    print("REQUESTED PLOTS COMPLETE")
    print("=" * 72)
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
