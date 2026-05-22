#!/usr/bin/env python3
"""
Hindi vs English negativity scatter using the updated final-stance dataset.

This version uses the same keyword exclusions as
FINAL_STANCE/requested_plots_no_low_metric_keywords and writes a paper-ready
PNG/PDF to both the legacy output folder and final_visualisations.

Exact input dataset copy used:
    scripts_copy/extracted_ALL_35keywords_english_hindi_only_final_with_existing_and_inferred_stance.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


REPO_DIR = Path("/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-")
SCRIPT_DIR = Path(__file__).parent.resolve()
FINAL_CSV = SCRIPT_DIR / "extracted_ALL_35keywords_english_hindi_only_final_with_existing_and_inferred_stance.csv"
OUTPUT_DIR = REPO_DIR / "final_results+visualisations_folder" / "codes" / "output"
FINAL_VIS_DIR = REPO_DIR / "final_visualisations"
REQUESTED_OUTPUT_DIR = (
    REPO_DIR
    / "Additional_Steps"
    / "Crosschecking_and_Joining"
    / "FINAL_STANCE"
    / "requested_plots_no_low_metric_keywords"
)
INPUT_DATASETS = {
    "final_stance_csv": FINAL_CSV,
}

VALID_STANCES = ["favor", "neutral", "against"]
VALID_PARTIES = ["pro ruling", "pro opposition"]
LANGUAGES = ["english", "hindi"]
MIN_ROWS_PER_PARTY_LANGUAGE = 20

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

DISPLAY_OVERRIDES = {
    "aatmanirbhar": "Aatmanirbhar",
    "caa": "CAA",
    "gdp": "GDP",
    "rahulgandhi": "Rahul Gandhi",
}

PARTY_LABELS = {
    "pro ruling": "Pro Ruling",
    "pro opposition": "Pro Opposition",
}


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
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


def load_updated_rows() -> pd.DataFrame:
    if not FINAL_CSV.exists():
        raise FileNotFoundError(f"Missing final stance CSV: {FINAL_CSV}")

    print("=" * 70)
    print("LOADING UPDATED FINAL STANCE DATA")
    print("=" * 70)
    print(f"CSV: {FINAL_CSV}")
    print("Excluded keywords:", ", ".join(display_keyword(k) for k in sorted(EXCLUDED_KEYWORDS)))

    usecols = [
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

    for chunk in pd.read_csv(FINAL_CSV, usecols=usecols, dtype="string", chunksize=250000, low_memory=False):
        total_rows += len(chunk)
        chunk["keyword"] = chunk["keyword"].fillna("").astype(str).str.lower().str.strip()
        chunk["stance"] = chunk["fewshot_label"].fillna("").astype(str).str.lower().str.strip()
        chunk["party"] = chunk["_label_norm"].fillna("").astype(str).str.lower().str.strip()
        chunk["language_group"] = derive_language_group(chunk)

        chunk = chunk[
            (chunk["keyword"] != "")
            & (chunk["stance"].isin(VALID_STANCES))
            & (chunk["party"].isin(VALID_PARTIES))
            & (chunk["language_group"].isin(LANGUAGES))
            & (~chunk["keyword"].isin(EXCLUDED_KEYWORDS))
        ].copy()
        if not chunk.empty:
            chunks.append(chunk[["keyword", "stance", "party", "language_group"]])
            kept_rows += len(chunk)

    if not chunks:
        raise ValueError("No rows remained after filtering.")

    df = pd.concat(chunks, ignore_index=True)
    print(f"Rows read: {total_rows:,}")
    print(f"Rows retained: {kept_rows:,}")
    print(f"Keywords retained: {df['keyword'].nunique()}")
    print("Language rows:")
    print(df["language_group"].value_counts().to_string())
    return df


def compute_negativity_points(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    rows: list[dict[str, float | int | str]] = []
    metrics_rows: list[dict[str, float | int | str]] = []

    for party in VALID_PARTIES:
        party_rows = []
        for keyword in sorted(df["keyword"].unique()):
            english = df[
                (df["keyword"] == keyword)
                & (df["party"] == party)
                & (df["language_group"] == "english")
            ]
            hindi = df[
                (df["keyword"] == keyword)
                & (df["party"] == party)
                & (df["language_group"] == "hindi")
            ]

            english_n = len(english)
            hindi_n = len(hindi)
            if english_n <= MIN_ROWS_PER_PARTY_LANGUAGE or hindi_n <= MIN_ROWS_PER_PARTY_LANGUAGE:
                continue

            english_against = (english["stance"] == "against").mean() * 100.0
            hindi_against = (hindi["stance"] == "against").mean() * 100.0
            party_rows.append(
                {
                    "keyword": keyword,
                    "party": party,
                    "english_against_pct": english_against,
                    "hindi_against_pct": hindi_against,
                    "english_rows": english_n,
                    "hindi_rows": hindi_n,
                    "difference_hindi_minus_english": hindi_against - english_against,
                }
            )

        party_df = pd.DataFrame(party_rows)
        if party_df.empty:
            metrics_rows.append({"party": PARTY_LABELS[party], "n_points": 0})
            continue

        perp = party_df["difference_hindi_minus_english"].to_numpy(dtype=float) / np.sqrt(2)
        metrics_rows.append(
            {
                "party": PARTY_LABELS[party],
                "n_points": len(party_df),
                "rmse": float(np.sqrt(np.mean(perp**2))),
                "mae": float(np.mean(np.abs(perp))),
                "median_abs": float(np.median(np.abs(perp))),
                "mean_signed": float(np.mean(perp)),
            }
        )
        rows.extend(party_rows)

    points = pd.DataFrame(rows)
    if points.empty:
        raise ValueError("No keyword-party-language points met the minimum row threshold.")
    return points, metrics_rows


def plot_negativity_scatter(points: pd.DataFrame, metrics_rows: list[dict[str, float]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_VIS_DIR.mkdir(parents=True, exist_ok=True)
    REQUESTED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    keyword_order = sorted(points["keyword"].unique().tolist())
    keyword_numbers = {keyword: i + 1 for i, keyword in enumerate(keyword_order)}
    points = points.copy()
    points["keyword_id"] = points["keyword"].map(keyword_numbers)
    points.to_csv(OUTPUT_DIR / "6_negativity_scatter_points.csv", index=False)

    max_axis = max(
        75,
        float(points[["english_against_pct", "hindi_against_pct"]].max().max()) + 7,
    )
    max_axis = min(100, np.ceil(max_axis / 5) * 5)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.55), facecolor="white", sharex=True, sharey=True)
    point_color = "#c0392b"

    for ax, party in zip(axes, VALID_PARTIES):
        party_points = points[points["party"] == party].copy()
        ax.set_title(PARTY_LABELS[party], pad=10)
        ax.plot([0, max_axis], [0, max_axis], "k--", alpha=0.55, linewidth=1.8)
        ax.fill_between([0, max_axis], [0, max_axis], [max_axis, max_axis], alpha=0.08, color="#c0392b")
        ax.fill_between([0, max_axis], [0, 0], [0, max_axis], alpha=0.08, color="#27ae60")

        if party_points.empty:
            ax.text(0.5, 0.5, "No qualifying keywords", transform=ax.transAxes, ha="center", va="center")
            continue

        ax.scatter(
            party_points["english_against_pct"],
            party_points["hindi_against_pct"],
            s=330,
            c=point_color,
            alpha=0.68,
            edgecolors="white",
            linewidth=1.3,
            zorder=3,
        )

        label_offsets = [
            (0, 0),
            (14, 10),
            (-14, 10),
            (14, -12),
            (-14, -12),
            (0, 16),
            (0, -17),
            (22, 0),
            (-22, 0),
        ]
        party_points = party_points.sort_values(["english_against_pct", "hindi_against_pct"]).reset_index(drop=True)
        for label_idx, (_, row) in enumerate(party_points.iterrows()):
            dx, dy = label_offsets[label_idx % len(label_offsets)]
            text = ax.annotate(
                str(int(row["keyword_id"])),
                xy=(row["english_against_pct"], row["hindi_against_pct"]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=5.8,
                color="black",
                zorder=4,
                bbox=dict(facecolor="white", edgecolor="#777777", boxstyle="circle,pad=0.18", alpha=0.94),
                arrowprops=(
                    dict(arrowstyle="-", color="#777777", linewidth=0.7, alpha=0.75, shrinkA=2, shrinkB=2)
                    if dx or dy
                    else None
                ),
            )
            text.set_path_effects([pe.withStroke(linewidth=1.7, foreground="white")])

        ax.set_xlim(0, max_axis)
        ax.set_ylim(0, max_axis)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.24)

    axes[0].set_ylabel('Hindi "Against" %')
    for ax in axes:
        ax.set_xlabel('English "Against" %')

    legend_text = "\n".join(f"{keyword_numbers[k]}: {display_keyword(k)}" for k in keyword_order)
    fig.text(
        0.835,
        0.83,
        "Keyword IDs\n" + legend_text,
        ha="left",
        va="top",
        fontsize=5.7,
        linespacing=0.94,
        bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.45"),
    )

    plt.subplots_adjust(left=0.075, right=0.80, top=0.90, bottom=0.105, wspace=0.13)

    output_paths = [
        OUTPUT_DIR / "6_negativity_scatter.png",
        FINAL_VIS_DIR / "6_negativity_scatter.png",
        REQUESTED_OUTPUT_DIR / "6_negativity_scatter.png",
    ]
    pdf_paths = [
        OUTPUT_DIR / "6_negativity_scatter.pdf",
        FINAL_VIS_DIR / "6_negativity_scatter.pdf",
        REQUESTED_OUTPUT_DIR / "6_negativity_scatter.pdf",
    ]
    for path in output_paths:
        fig.savefig(path, dpi=600, bbox_inches="tight")
        print(f"Saved: {path}")
    for path in pdf_paths:
        fig.savefig(path, bbox_inches="tight", format="pdf")
        print(f"Saved: {path}")
    plt.close(fig)

    metrics_path = OUTPUT_DIR / "6_negativity_scatter_metrics.txt"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write("Updated final-stance dataset with requested low-quality keyword exclusions.\n")
        f.write("Distances are perpendicular to the diagonal y=x, in percentage points.\n")
        f.write(f"Minimum rows per party-language cell: > {MIN_ROWS_PER_PARTY_LANGUAGE}\n\n")
        for row in metrics_rows:
            f.write(f"{row['party']} (n={row['n_points']} keywords)\n")
            if row["n_points"] == 0:
                f.write("  No data available\n\n")
                continue
            f.write(f"  RMSE: {row['rmse']:.3f}\n")
            f.write(f"  MAE: {row['mae']:.3f}\n")
            f.write(f"  MedianAbs: {row['median_abs']:.3f}\n")
            f.write(f"  MeanSigned: {row['mean_signed']:.3f}\n\n")
    print(f"Saved: {metrics_path}")


def main() -> None:
    print("=" * 70)
    print("UPDATED HINDI VS ENGLISH NEGATIVITY SCATTER")
    print("=" * 70)
    df = load_updated_rows()
    points, metrics_rows = compute_negativity_points(df)
    print(f"Qualifying plotted points: {len(points)}")
    print(f"Qualifying unique keywords: {points['keyword'].nunique()}")
    plot_negativity_scatter(points, metrics_rows)


if __name__ == "__main__":
    main()
