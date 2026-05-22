#!/usr/bin/env python3
"""
Language Divergence Scatter Plot (English vs Hindi)
===================================================
Compares stance divergence across English and Hindi in the final merged stance
CSV.

Each keyword contributes two points:
- English
- Hindi

For each language point:
- x = favor difference   = favor%(pro ruling) - favor%(pro opposition)
- y = against difference = against%(pro ruling) - against%(pro opposition)

The plot connects the English and Hindi points for each keyword and highlights
the keywords with the largest English-vs-Hindi separation directly on the
figure.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")


BASE_DIR = Path(__file__).parent.resolve()
CSV_PATH = BASE_DIR / "extracted_ALL_35keywords_english_hindi_only_final_with_existing_and_inferred_stance.csv"
OUTPUT_PNG = BASE_DIR / "divergence_scatter_english_hindi.png"
OUTPUT_PDF = BASE_DIR / "divergence_scatter_english_hindi.pdf"

LANGUAGES = ["english", "hindi"]
PARTIES = ["pro ruling", "pro opposition"]
MIN_UNIQUE_PER_PARTY = 2
HIGHLIGHT_TOP_N = 5

LANG_MARKERS = {"english": "o", "hindi": "s"}
LANG_LABELS = {"english": "English", "hindi": "Hindi"}
LANG_TEXT_OFFSETS = {"english": (-0.8, 0.9), "hindi": (0.8, -0.9)}

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


def derive_language_group(chunk: pd.DataFrame) -> pd.Series:
    language = chunk["language"].fillna("").astype(str).str.lower().str.strip()
    translation_route = chunk["translation_route"].fillna("").astype(str).str.lower().str.strip()
    lid_lang = chunk["lid_lang"].fillna("").astype(str).str.lower().str.strip()

    group = language.copy()
    hinglish_mask = (translation_route == "hinglish_roman") | (lid_lang == "hin_latn")
    group.loc[hinglish_mask] = "hinglish"
    return group


def load_data() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    print("=" * 70)
    print("LOADING FINAL STANCE DATA (ENGLISH vs HINDI)")
    print("=" * 70)
    print(f"CSV Path: {CSV_PATH}")

    chunks = []
    for i, chunk in enumerate(pd.read_csv(CSV_PATH, chunksize=100000, low_memory=False)):
        chunk = chunk[chunk["fewshot_label"].isin(["favor", "against", "neutral"])].copy()
        chunk["stance"] = chunk["fewshot_label"].astype(str).str.lower().str.strip()
        chunk["party"] = chunk["_label_norm"].astype(str).str.lower().str.strip()
        chunk["keyword"] = chunk["keyword"].astype(str).str.lower().str.strip()
        chunk["language_group"] = derive_language_group(chunk)
        chunk = chunk[chunk["party"].isin(PARTIES)]
        chunk = chunk[chunk["language_group"].isin(LANGUAGES)]
        chunks.append(chunk)
        if (i + 1) % 5 == 0:
            print(f"  Processed {(i + 1) * 100000:,} rows...")

    df = pd.concat(chunks, ignore_index=True)

    print("\nLanguage distribution")
    for lang in LANGUAGES:
        subset = df[df["language_group"] == lang]
        print(
            f"  {LANG_LABELS[lang]}: {len(subset):,} rows | "
            f"{subset['source_row'].nunique():,} unique tweets"
        )
    print("-" * 70)
    return df


def compute_language_points(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows = []

    for keyword in sorted(df["keyword"].unique()):
        keyword_df = df[df["keyword"] == keyword]
        qualifies = True

        for language in LANGUAGES:
            for party in PARTIES:
                subset = keyword_df[
                    (keyword_df["language_group"] == language) & (keyword_df["party"] == party)
                ]
                if subset["source_row"].nunique() < MIN_UNIQUE_PER_PARTY:
                    qualifies = False

        if not qualifies:
            continue

        for language in LANGUAGES:
            ruling = keyword_df[
                (keyword_df["language_group"] == language) & (keyword_df["party"] == "pro ruling")
            ]
            opposition = keyword_df[
                (keyword_df["language_group"] == language) & (keyword_df["party"] == "pro opposition")
            ]

            ruling_pcts = ruling["stance"].value_counts(normalize=True) * 100
            opposition_pcts = opposition["stance"].value_counts(normalize=True) * 100

            rows.append(
                {
                    "keyword": keyword,
                    "language_group": language,
                    "favor_diff": ruling_pcts.get("favor", 0) - opposition_pcts.get("favor", 0),
                    "against_diff": ruling_pcts.get("against", 0) - opposition_pcts.get("against", 0),
                    "ruling_n": ruling["source_row"].nunique(),
                    "opposition_n": opposition["source_row"].nunique(),
                    "total_n": ruling["source_row"].nunique() + opposition["source_row"].nunique(),
                }
            )

    points = pd.DataFrame(rows)
    qualifying_keywords = sorted(points["keyword"].unique()) if not points.empty else []
    return points, qualifying_keywords


def compute_keyword_spread(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keyword, group in points.groupby("keyword"):
        english = group[group["language_group"] == "english"].iloc[0]
        hindi = group[group["language_group"] == "hindi"].iloc[0]
        dist = math.dist(
            (english["favor_diff"], english["against_diff"]),
            (hindi["favor_diff"], hindi["against_diff"]),
        )
        rows.append(
            {
                "keyword": keyword,
                "distance": dist,
                "centroid_x": group["favor_diff"].mean(),
                "centroid_y": group["against_diff"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("distance", ascending=False)


def plot_language_divergence(points: pd.DataFrame, spread_df: pd.DataFrame) -> None:
    if points.empty:
        raise ValueError("No keywords qualify for the English-Hindi comparison.")

    fig, ax = plt.subplots(figsize=(16, 12))

    keyword_list = sorted(points["keyword"].unique())
    cmap = plt.cm.get_cmap("tab20", len(keyword_list))
    keyword_colors = {keyword: cmap(i) for i, keyword in enumerate(keyword_list)}
    keyword_numbers = {keyword: idx + 1 for idx, keyword in enumerate(keyword_list)}
    highlight_keywords = set(spread_df.head(HIGHLIGHT_TOP_N)["keyword"])

    x_vals = points["favor_diff"].to_numpy()
    y_vals = points["against_diff"].to_numpy()
    x_pad = max(8.0, (x_vals.max() - x_vals.min()) * 0.16 if len(x_vals) else 8.0)
    y_pad = max(8.0, (y_vals.max() - y_vals.min()) * 0.16 if len(y_vals) else 8.0)

    ax.axhline(0, color="black", linestyle="-", alpha=0.25, zorder=1)
    ax.axvline(0, color="black", linestyle="-", alpha=0.25, zorder=1)
    ax.set_xlim(x_vals.min() - x_pad, x_vals.max() + x_pad)
    ax.set_ylim(y_vals.min() - y_pad, y_vals.max() + y_pad)

    for keyword, group in points.groupby("keyword"):
        color = keyword_colors[keyword]
        group = group.set_index("language_group").loc[LANGUAGES].reset_index()
        is_highlight = keyword in highlight_keywords

        ax.plot(
            group["favor_diff"].to_numpy(),
            group["against_diff"].to_numpy(),
            color=color,
            linewidth=3.2 if is_highlight else 1.4,
            alpha=0.95 if is_highlight else 0.4,
            zorder=2,
        )

        for _, row in group.iterrows():
            lang = row["language_group"]
            size = np.clip(np.sqrt(row["total_n"]) * 125, 130, 2200)
            ax.scatter(
                row["favor_diff"],
                row["against_diff"],
                s=size,
                marker=LANG_MARKERS[lang],
                color=color,
                edgecolors="#111111",
                linewidth=1.5 if is_highlight else 0.8,
                alpha=0.92,
                zorder=3,
            )

            dx, dy = LANG_TEXT_OFFSETS[lang]
            txt = ax.text(
                row["favor_diff"] + dx,
                row["against_diff"] + dy,
                str(keyword_numbers[keyword]),
                fontsize=10,
                ha="center",
                va="center",
                zorder=4,
                bbox=dict(
                    facecolor="white",
                    edgecolor=color if is_highlight else "#999999",
                    boxstyle="circle,pad=0.20",
                    alpha=0.95,
                ),
            )
            txt.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])

    q_box = dict(boxstyle="round4,pad=0.45", fc="#f8f9fa", ec="#b2bec3", lw=1.2, alpha=0.95)
    ax.text(
        0.02,
        0.98,
        "Opposition Favors\nRuling Against",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="#d63031",
        bbox=q_box,
    )
    ax.text(
        0.98,
        0.98,
        "Ruling High Polarization",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="#e67e22",
        bbox=q_box,
    )
    ax.text(
        0.02,
        0.02,
        "Opposition High Polarization",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#0984e3",
        bbox=q_box,
    )
    ax.text(
        0.98,
        0.02,
        "Ruling Favors\nOpposition Against",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="#27ae60",
        bbox=q_box,
    )

    for _, row in spread_df.head(HIGHLIGHT_TOP_N).iterrows():
        ax.annotate(
            f"{keyword_numbers[row['keyword']]} {row['keyword'].title()}\nE-H gap={row['distance']:.1f}",
            xy=(row["centroid_x"], row["centroid_y"]),
            xytext=(row["centroid_x"] + 3.5, row["centroid_y"] + 3.5),
            fontsize=10,
            ha="left",
            va="bottom",
            bbox=dict(facecolor="#eef7ff", edgecolor="#1f77b4", boxstyle="round,pad=0.35", alpha=0.98),
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.4),
            zorder=5,
        )

    note = (
        f"One point each for English and Hindi per keyword. "
        f"Keywords shown: {len(keyword_list)}.\n"
        f"Highlighted keywords have the largest English-Hindi separation in divergence space."
    )
    ax.text(
        0.5,
        1.02,
        note,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.35"),
    )

    ax.set_xlabel("Favor Difference (Ruling % - Opposition %)")
    ax.set_ylabel("Against Difference (Ruling % - Opposition %)")

    language_handles = [
        Line2D(
            [0],
            [0],
            marker=LANG_MARKERS[lang],
            color="white",
            label=LANG_LABELS[lang],
            markerfacecolor="#555555",
            markeredgecolor="#111111",
            markersize=10,
            linewidth=0,
        )
        for lang in LANGUAGES
    ]
    lang_legend = ax.legend(handles=language_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
    ax.add_artist(lang_legend)

    legend_keywords = "\n".join(f"{keyword_numbers[k]}: {k.title()}" for k in keyword_list)
    fig.text(
        0.855,
        0.58,
        "Keyword IDs\n" + legend_keywords,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.45"),
        family="serif",
    )

    spread_lines = [
        f"{keyword_numbers[row.keyword]} {row.keyword.title()}: {row.distance:.1f}"
        for _, row in spread_df.head(HIGHLIGHT_TOP_N).iterrows()
    ]
    fig.text(
        0.855,
        0.24,
        "Largest English-Hindi Gaps\n" + "\n".join(spread_lines),
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="#eef7ff", edgecolor="#1f77b4", boxstyle="round,pad=0.45"),
        family="serif",
    )

    plt.subplots_adjust(left=0.08, right=0.80, top=0.90, bottom=0.10)
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight", format="pdf")
    plt.close()

    print("\nTop highlighted differences:")
    for _, row in spread_df.head(HIGHLIGHT_TOP_N).iterrows():
        print(f"  {row['keyword'].title()}: E-H gap={row['distance']:.2f}")
    print(f"\nPlot saved to: {OUTPUT_PNG}")
    print(f"Plot saved to: {OUTPUT_PDF}")


def main() -> None:
    print("=" * 70)
    print("LANGUAGE DIVERGENCE SCATTER (ENGLISH vs HINDI)")
    print("=" * 70)

    df = load_data()
    points, qualifying_keywords = compute_language_points(df)
    if not qualifying_keywords:
        raise SystemExit("No keywords qualify for the English-Hindi comparison.")

    print(
        f"Qualifying keywords with >= {MIN_UNIQUE_PER_PARTY} unique tweets "
        f"per party-language cell: {len(qualifying_keywords)}"
    )
    spread_df = compute_keyword_spread(points)
    plot_language_divergence(points, spread_df)

    print("=" * 70)
    print("COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
