"""
Stance stack plots for temporal LoRA predictions.
Saves all figures to runs/.../evaluation/stance_stack_plots/
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
RUN_DIR = Path(
    "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-"
    "/codes/4_finetuning/4_f_temporal/runs"
    "/temporal_lora_2020_2021_test_2022_2023_gpus_5_7"
)
PRED_CSV = RUN_DIR / "predictions" / "temporal_mistral_lora_predictions.csv"
OUT_DIR  = RUN_DIR / "evaluation" / "stance_stack_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PRED_CSV)

# ── shared config ──────────────────────────────────────────────────────────────
STANCE_ORDER  = ["For", "Neutral", "Against"]
STANCE_COLORS = {"For": "#2196F3", "Neutral": "#9E9E9E", "Against": "#F44336"}

PARTISAN_ORDER  = ["Pro Ruling", "Pro OPP"]
PARTISAN_COLORS = {"Pro Ruling": "#FF9800", "Pro OPP": "#673AB7"}

def stacked_bar(ax, counts_df, stance_order, colors, title, xlabel="", ylabel="Count"):
    """Draw a horizontal stacked bar chart from a (categories × stances) DataFrame."""
    categories = counts_df.index.tolist()
    lefts = np.zeros(len(categories))
    for stance in stance_order:
        if stance not in counts_df.columns:
            continue
        vals = counts_df[stance].values
        bars = ax.barh(categories, vals, left=lefts,
                       color=colors[stance], label=stance, edgecolor="white", linewidth=0.4)
        # label bars that are wide enough
        for bar, v in zip(bars, vals):
            if v > 0:
                bx = bar.get_x() + bar.get_width() / 2
                by = bar.get_y() + bar.get_height() / 2
                if bar.get_width() >= 2:
                    ax.text(bx, by, str(int(v)), ha="center", va="center",
                            fontsize=7, color="white", fontweight="bold")
        lefts += vals
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel or "Count", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(title="Stance", loc="lower right", fontsize=8, title_fontsize=8)
    ax.tick_params(labelsize=8)
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    return ax


def pivot_counts(data, group_col, stance_col="stance"):
    ct = (data.groupby([group_col, stance_col])
               .size()
               .unstack(fill_value=0)
               .reindex(columns=STANCE_ORDER, fill_value=0))
    # sort by total count descending
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=True).index]
    return ct


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Ground-truth stance by keyword
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 12))
ct = pivot_counts(df, "keyword", "stance")
stacked_bar(ax, ct, STANCE_ORDER, STANCE_COLORS,
            "Ground-truth Stance by Keyword  (2022–2023 test set)")
plt.tight_layout()
fig.savefig(OUT_DIR / "1_gt_stance_by_keyword.png", dpi=150)
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 2.  Predicted stance by keyword
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 12))
ct = pivot_counts(df, "keyword", "lora_stance")
stacked_bar(ax, ct, STANCE_ORDER, STANCE_COLORS,
            "LoRA-predicted Stance by Keyword  (2022–2023 test set)")
plt.tight_layout()
fig.savefig(OUT_DIR / "2_pred_stance_by_keyword.png", dpi=150)
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 3.  Ground-truth vs predicted — side-by-side per keyword (proportions)
# ══════════════════════════════════════════════════════════════════════════════
def prop_pivot(data, group_col, stance_col):
    ct = (data.groupby([group_col, stance_col])
               .size()
               .unstack(fill_value=0)
               .reindex(columns=STANCE_ORDER, fill_value=0))
    return ct.div(ct.sum(axis=1), axis=0)

gt_prop   = prop_pivot(df, "keyword", "stance")
pred_prop = prop_pivot(df, "keyword", "lora_stance")

# sort keywords by gt Against proportion (most "against" at top)
order = gt_prop["Against"].sort_values(ascending=True).index

fig, axes = plt.subplots(1, 2, figsize=(16, 12), sharey=True)
for ax, prop, title in zip(axes,
                            [gt_prop.loc[order], pred_prop.loc[order]],
                            ["Ground Truth (proportion)", "LoRA Predicted (proportion)"]):
    lefts = np.zeros(len(order))
    for stance in STANCE_ORDER:
        vals = prop[stance].values
        ax.barh(order, vals, left=lefts,
                color=STANCE_COLORS[stance], label=stance,
                edgecolor="white", linewidth=0.3)
        lefts += vals
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Proportion", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.legend(title="Stance", fontsize=8, title_fontsize=8, loc="lower right")

fig.suptitle("Stance Proportion by Keyword: Ground Truth vs LoRA Predicted",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(OUT_DIR / "3_gt_vs_pred_proportion_by_keyword.png", dpi=150, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 4.  Stance by partisan lean  (ground truth vs predicted)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, stance_col, title in zip(axes,
                                  ["stance", "lora_stance"],
                                  ["Ground Truth", "LoRA Predicted"]):
    ct = (df.groupby(["tweet_label", stance_col])
             .size()
             .unstack(fill_value=0)
             .reindex(index=PARTISAN_ORDER, columns=STANCE_ORDER, fill_value=0))
    stacked_bar(ax, ct, STANCE_ORDER, STANCE_COLORS, title,
                xlabel="Count", ylabel="Partisan lean")

fig.suptitle("Stance by Partisan Lean", fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT_DIR / "4_stance_by_partisan_lean.png", dpi=150)
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 5.  Stance by year  (ground truth vs predicted)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, stance_col, title in zip(axes,
                                  ["stance", "lora_stance"],
                                  ["Ground Truth", "LoRA Predicted"]):
    ct = (df.groupby(["year", stance_col])
             .size()
             .unstack(fill_value=0)
             .reindex(columns=STANCE_ORDER, fill_value=0))
    stacked_bar(ax, ct, STANCE_ORDER, STANCE_COLORS, title,
                xlabel="Count", ylabel="Year")

fig.suptitle("Stance by Year", fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT_DIR / "5_stance_by_year.png", dpi=150)
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 6.  Stance by keyword × year  (ground truth, two panels)
# ══════════════════════════════════════════════════════════════════════════════
for stance_col, fname_tag, title_tag in [
        ("stance",      "gt",   "Ground Truth"),
        ("lora_stance", "pred", "LoRA Predicted"),
]:
    fig, axes = plt.subplots(1, 2, figsize=(18, 12), sharey=False)
    for ax, yr in zip(axes, [2022, 2023]):
        sub = df[df["year"] == yr]
        ct  = pivot_counts(sub, "keyword", stance_col)
        stacked_bar(ax, ct, STANCE_ORDER, STANCE_COLORS,
                    f"{title_tag}  —  {yr}",
                    xlabel="Count", ylabel="Keyword")
    fig.suptitle(f"Stance by Keyword (per year) — {title_tag}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"6_stance_keyword_year_{fname_tag}.png", dpi=150)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 7.  Stance by partisan lean × keyword  (ground truth proportion heatmap)
# ══════════════════════════════════════════════════════════════════════════════
for stance_col, fname_tag, title_tag in [
        ("stance",      "gt",   "Ground Truth"),
        ("lora_stance", "pred", "LoRA Predicted"),
]:
    fig, axes = plt.subplots(1, 2, figsize=(18, 12), sharey=False)
    for ax, partisan in zip(axes, PARTISAN_ORDER):
        sub = df[df["tweet_label"] == partisan]
        ct  = pivot_counts(sub, "keyword", stance_col)
        stacked_bar(ax, ct, STANCE_ORDER, STANCE_COLORS,
                    f"{title_tag}  —  {partisan}",
                    xlabel="Count", ylabel="Keyword")
    fig.suptitle(f"Stance by Keyword × Partisan Lean — {title_tag}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"7_stance_keyword_partisan_{fname_tag}.png", dpi=150)
    plt.close()

print(f"All plots saved to:\n  {OUT_DIR}")
for p in sorted(OUT_DIR.iterdir()):
    print(f"  {p.name}")
