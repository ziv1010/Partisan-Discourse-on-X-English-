#!/usr/bin/env python3
"""
Normalized Divergence Scatter Plot (English + Hindi Combined)
=============================================================
Creates a publication-quality divergence scatter plot for stance analysis.
Uses combined stance results containing both English and Hindi tweets.

Key features:
- Uses combined_stance_results.csv (English + Hindi merged data)
- Counts unique tweets using source_row identifier
- Displays numbered labels with legend
- Shows language distribution stats for verification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patheffects as pe
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output" / "divergence_scatter"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Combined CSV path (contains both English and Hindi data)
CSV_PATH = BASE_DIR.parent / "combined_stance_results.csv"

plt.style.use("seaborn-v0_8-whitegrid")

PARTY_COLORS = {"pro ruling": "#FF6B35", "pro opposition": "#004E89"}
FS_TITLE, FS_LABEL, FS_ANNOT, FS_QUAD = 16, 12, 9, 10


# ==========================================
# 2. DATA LOADING (with verification)
# ==========================================
def load_data():
    """Load combined stance data and verify both languages are included."""
    if not Path(CSV_PATH).exists():
        print(f"ERROR: CSV not found at {CSV_PATH}")
        return pd.DataFrame()
    
    print("=" * 70)
    print("LOADING COMBINED STANCE DATA (ENGLISH + HINDI)")
    print("=" * 70)
    print(f"CSV Path: {CSV_PATH}")
    
    # Read in chunks to handle large file
    chunks = []
    chunk_iter = pd.read_csv(CSV_PATH, chunksize=100000, low_memory=False)
    
    for i, chunk in enumerate(chunk_iter):
        # Filter to valid stances
        chunk = chunk[chunk["fewshot_label"].isin(["favor", "against", "neutral"])].copy()
        chunk["stance"] = chunk["fewshot_label"]
        chunk["party"] = chunk["_label_norm"].str.lower().str.strip()
        chunk["keyword"] = chunk["keyword"].astype(str).str.lower().str.strip()
        chunk["language"] = chunk["language"].str.lower().str.strip() if "language" in chunk.columns else "unknown"
        
        # Filter to valid parties
        chunk = chunk[chunk["party"].isin(["pro ruling", "pro opposition"])]
        chunks.append(chunk)
        
        if (i + 1) % 5 == 0:
            print(f"  Processed {(i+1) * 100000:,} rows...")
    
    df = pd.concat(chunks, ignore_index=True)
    
    # ===== VERIFICATION: Language distribution =====
    print("\n" + "-" * 50)
    print("LANGUAGE VERIFICATION")
    print("-" * 50)
    
    lang_counts = df["language"].value_counts()
    print(f"Total rows: {len(df):,}")
    for lang, count in lang_counts.items():
        print(f"  {lang.capitalize()}: {count:,} rows ({count/len(df)*100:.1f}%)")
    
    # Count unique tweets
    unique_tweets = df["source_row"].nunique()
    print(f"\nUnique tweets (source_row): {unique_tweets:,}")
    
    # Unique tweets by language
    for lang in df["language"].unique():
        lang_df = df[df["language"] == lang]
        unique_lang = lang_df["source_row"].nunique()
        print(f"  {lang.capitalize()} unique tweets: {unique_lang:,}")
    
    print("-" * 50)
    
    return df


# ==========================================
# 3. NUMBERED SCATTER PLOT
# ==========================================
def plot_divergence_scatter(df):
    """Create divergence scatter plot with numbered keyword labels."""
    if df.empty:
        print("ERROR: No data to plot!")
        return

    # ---- compute per keyword stats (using unique tweets) ----
    print("\nComputing per-keyword statistics...")
    res = []
    for kw in df["keyword"].unique():
        for p in ["pro ruling", "pro opposition"]:
            sub = df[(df["keyword"] == kw) & (df["party"] == p)]
            if len(sub) < 10:
                continue
            
            # Count unique tweets for this keyword+party combination
            n_unique = sub["source_row"].nunique()
            
            # Calculate stance percentages
            pcts = sub["stance"].value_counts(normalize=True) * 100
            
            res.append(
                {
                    "keyword": kw,
                    "party": p,
                    "favor": pcts.get("favor", 0),
                    "against": pcts.get("against", 0),
                    "n": n_unique,  # Using unique tweet count
                    "n_rows": len(sub),  # Row count for comparison
                }
            )

    pdf = pd.DataFrame(res)
    if pdf.empty:
        print("ERROR: No valid keyword data after filtering!")
        return

    # Pivot to get differences
    pivot = pdf.pivot(index="keyword", columns="party", values=["favor", "against", "n"]).dropna()

    fav_diff = (pivot["favor"]["pro ruling"] - pivot["favor"]["pro opposition"]).values.astype(float)
    agn_diff = (pivot["against"]["pro ruling"] - pivot["against"]["pro opposition"]).values.astype(float)
    total_n = (pivot["n"]["pro ruling"] + pivot["n"]["pro opposition"]).values.astype(float)
    keywords = pivot.index.tolist()

    print(f"\nKeywords included in plot: {len(keywords)}")
    print(f"Total unique tweets across all keywords: {int(total_n.sum()):,}")

    # ---- STABLE bubble sizing ----
    s_min, s_max = 200, 24000
    sizes = np.sqrt(total_n) * 120
    sizes = np.clip(sizes, s_min, s_max)

    # ---- deterministic ordering (alphabetical) ----
    order = np.argsort(np.array(keywords))
    fav_diff, agn_diff, total_n, sizes = fav_diff[order], agn_diff[order], total_n[order], sizes[order]
    keywords = [keywords[i] for i in order]

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(16, 12))

    scatter = ax.scatter(
        fav_diff,
        agn_diff,
        s=sizes,
        alpha=0.45,
        c=fav_diff,
        cmap="RdYlGn",
        edgecolors="k",
        linewidth=0.6,
        zorder=2,
    )

    # axis lines
    ax.axhline(0, color="black", linestyle="-", alpha=0.25, zorder=1)
    ax.axvline(0, color="black", linestyle="-", alpha=0.25, zorder=1)

    # padding
    ax.set_xlim(np.min(fav_diff) - 25, np.max(fav_diff) + 25)
    ax.set_ylim(np.min(agn_diff) - 25, np.max(agn_diff) + 25)

    # ---- Number labels with offsets ----
    offsets = [
        (0.0, 0.0),
        (1.2, 0.0),
        (-1.2, 0.0),
        (0.0, 1.2),
        (0.0, -1.2),
        (1.0, 1.0),
        (-1.0, 1.0),
        (1.0, -1.0),
        (-1.0, -1.0),
    ]

    legend_map = {}
    for i, kw in enumerate(keywords):
        label_num = i + 1
        legend_map[label_num] = kw

        dx, dy = offsets[i % len(offsets)]
        x, y = fav_diff[i] + dx, agn_diff[i] + dy

        txt = ax.text(
            x,
            y,
            str(label_num),
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            zorder=4,
            bbox=dict(facecolor="white", alpha=0.92, edgecolor="black", boxstyle="circle,pad=0.28"),
        )
        txt.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])

    # ---- quadrant labels ----
    q_box = dict(boxstyle="round4,pad=0.5", fc="#f8f9fa", ec="#b2bec3", lw=1.5, alpha=0.9)

    ax.text(
        0.02,
        0.98,
        "OPPOSITION FAVORS\nRULING AGAINST",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="black",
        fontsize=FS_QUAD,
        color="#d63031",
        bbox=q_box,
    )
    ax.text(
        0.98,
        0.98,
        "RULING HIGH POLARIZATION\n(High Favor & Against)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontweight="black",
        fontsize=FS_QUAD,
        color="#e67e22",
        bbox=q_box,
    )
    ax.text(
        0.02,
        0.02,
        "OPP. HIGH POLARIZATION\n(High Favor & Against)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontweight="black",
        fontsize=FS_QUAD,
        color="#0984e3",
        bbox=q_box,
    )
    ax.text(
        0.98,
        0.02,
        "RULING FAVORS\nOPPOSITION AGAINST",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontweight="black",
        fontsize=FS_QUAD,
        color="#27ae60",
        bbox=q_box,
    )

    ax.set_xlabel("Favor Difference (Ruling % - Opp %)", fontweight="bold", fontsize=FS_LABEL)
    ax.set_ylabel("Against Difference (Ruling % - Opp %)", fontweight="bold", fontsize=FS_LABEL)
    
    # ---- Get language stats for subtitle ----
    lang_counts = df["language"].value_counts()
    en_count = lang_counts.get("english", 0)
    hi_count = lang_counts.get("hindi", 0)
    
    ax.set_title(
        f"Normalized Stance Divergence Scatter (English + Hindi Combined)\n"
        f"English: {en_count:,} tweets | Hindi: {hi_count:,} tweets | Total: {len(df):,}",
        fontweight="bold",
        fontsize=FS_TITLE,
        pad=20
    )

    # ---- Legend outside (sorted by number, in 2 columns) ----
    items = [f"{k}: {v}" for k, v in sorted(legend_map.items(), key=lambda x: x[0])]
    midpoint = (len(items) + 1) // 2
    col1 = "\n".join(items[:midpoint])
    col2 = "\n".join(items[midpoint:])

    # Make room for legend + colorbar
    plt.subplots_adjust(right=0.78)

    fig.text(0.81, 0.86, col1, fontsize=9, va="top", bbox=dict(facecolor="white", alpha=0.9, edgecolor="black"))
    fig.text(0.91, 0.86, col2, fontsize=9, va="top", bbox=dict(facecolor="white", alpha=0.9, edgecolor="black"))

    cbar = plt.colorbar(scatter, pad=0.02)
    cbar.set_label("Favor Divergence Score", fontsize=FS_LABEL, fontweight="bold")

    outpath = OUTPUT_DIR / "divergence_scatter_english_hindi_combined.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nPlot saved to: {outpath}")
    return outpath


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DIVERGENCE SCATTER PLOT - ENGLISH + HINDI COMBINED")
    print("=" * 70)
    
    df_main = load_data()
    
    if not df_main.empty:
        plot_divergence_scatter(df_main)
        
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
