"""
Polarity Stability Check
Replicates the temporal stability analysis:
  - Splits influencers into period 2020-2021 vs 2022-2023
  - Computes per-period polarity by summing raw engagement counts
  - Reports Pearson r, Spearman rho, median absolute change, % same direction
  - Produces a scatter plot and aggregate trend table
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

CSV_PATH = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/1_cleaning_and_formatting/output/yearly_engagements_and_polarity.csv"
OUT_FIG   = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/1_cleaning_and_formatting/output/polarity_stability.png"

# ── load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df["year"] = df["year"].astype(int)

print(f"Total rows: {len(df)}")
print(f"Unique influencers: {df['original_author'].nunique()}")
print(f"Years: {sorted(df['year'].unique())}\n")

# ── aggregate counts by period ────────────────────────────────────────────────
def period_label(y):
    return "2020-2021" if y <= 2021 else "2022-2023"

df["period"] = df["year"].map(period_label)

period_agg = (
    df.groupby(["original_author", "period"])[["opposition", "ruling"]]
    .sum()
    .reset_index()
)

# Polarity formula: (ruling - opposition) / (ruling + opposition)
denom = period_agg["ruling"] + period_agg["opposition"]
period_agg["polarity"] = np.where(
    denom > 0,
    (period_agg["ruling"] - period_agg["opposition"]) / denom,
    np.nan,
)
period_agg = period_agg.dropna(subset=["polarity"])

# ── pivot to wide: one row per influencer, two polarity columns ───────────────
wide = period_agg.pivot(index="original_author", columns="period", values="polarity")
wide.columns.name = None
wide = wide.rename(columns={"2020-2021": "pol_early", "2022-2023": "pol_late"})

both_periods = wide.dropna(subset=["pol_early", "pol_late"])
only_one     = wide[wide.isna().any(axis=1)]

print(f"Influencers in BOTH periods : {len(both_periods)}")
print(f"Influencers in ONE period   : {len(only_one)}")
print(f"Total filtered influencers  : {len(wide)}\n")

early = both_periods["pol_early"].values
late  = both_periods["pol_late"].values

# ── correlation stats ─────────────────────────────────────────────────────────
pearson_r, pearson_p  = stats.pearsonr(early, late)
spearman_r, spearman_p = stats.spearmanr(early, late)
median_abs_change     = np.median(np.abs(late - early))
same_direction        = np.mean(np.sign(early) == np.sign(late)) * 100

print("─── Stability Results ────────────────────────────────────")
print(f"  Pearson  r  = {pearson_r:.3f}  (p={pearson_p:.2e})")
print(f"  Spearman ρ  = {spearman_r:.3f}  (p={spearman_p:.2e})")
print(f"  Median |Δpolarity|  = {median_abs_change:.3f}")
print(f"  Same directional alignment = {same_direction:.1f}%")
print("──────────────────────────────────────────────────────────\n")

# ── aggregate yearly trend ────────────────────────────────────────────────────
yearly_agg = (
    df.groupby("year")[["opposition", "ruling"]]
    .sum()
    .reset_index()
)
denom_y = yearly_agg["ruling"] + yearly_agg["opposition"]
yearly_agg["polarity_aggregate"] = (yearly_agg["ruling"] - yearly_agg["opposition"]) / denom_y
yearly_agg["n_influencers"] = df.groupby("year")["original_author"].nunique().values

print("─── Aggregate Polarity by Year ───────────────────────────")
print(yearly_agg[["year", "n_influencers", "opposition", "ruling", "polarity_aggregate"]].to_string(index=False))
print("──────────────────────────────────────────────────────────\n")

# Per-year median polarity across influencers
yearly_median = df.groupby("year")["polarity_yearly"].agg(["median", "mean", "std"]).reset_index()
yearly_median.columns = ["year", "median_polarity", "mean_polarity", "std_polarity"]
print("─── Per-Influencer Median Polarity by Year ───────────────")
print(yearly_median.to_string(index=False))
print("──────────────────────────────────────────────────────────\n")

# ── plot: violin per year with median trend overlaid ─────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

COLORS = {"2020": "#b2d8e8", "2021": "#b2d8e8", "2022": "#f4a582", "2023": "#f4a582"}
yrs = yearly_median["year"].values

# Violin for each year
yearly_groups = [df[df["year"] == y]["polarity_yearly"].dropna().values for y in yrs]
parts = ax.violinplot(yearly_groups, positions=yrs, widths=0.6,
                      showmedians=False, showextrema=False)
for i, (body, yr) in enumerate(zip(parts["bodies"], yrs)):
    body.set_facecolor(COLORS[str(yr)])
    body.set_edgecolor("white")
    body.set_alpha(0.75)

# IQR box (25–75) and median tick
q25 = [np.percentile(g, 25) for g in yearly_groups]
q75 = [np.percentile(g, 75) for g in yearly_groups]
meds = yearly_median["median_polarity"].values
ax.vlines(yrs, q25, q75, color="#333333", lw=4, zorder=3)
ax.scatter(yrs, meds, color="white", edgecolors="#333333", s=50, zorder=4, lw=1.5)

# Median trend line
ax.plot(yrs, meds, "-", color="#333333", lw=1.2, zorder=3)

# Period-level annotation (stability stats)
ax.axvspan(2019.6, 2021.4, color="#b2d8e8", alpha=0.12, zorder=0)
ax.axvspan(2021.6, 2023.4, color="#f4a582", alpha=0.12, zorder=0)
ax.text(2020.5, 1.08, "2020–2021", ha="center", fontsize=8, color="#2166ac")
ax.text(2022.5, 1.08, "2022–2023", ha="center", fontsize=8, color="#b2182b")

ax.axhline(0, color="gray", lw=0.6, ls="--")
ax.set_xticks(yrs)
ax.set_xlim(2019.6, 2023.4)
ax.set_ylim(-1.15, 1.15)
ax.set_xlabel("Year", fontsize=10)
ax.set_ylabel("Influencer Polarity\n(−1 = opposition, +1 = ruling)", fontsize=10)
ax.set_title("Aggregate Influencer Polarity Over Time", fontsize=11, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)

# Stats box anchored outside the axes, below the plot
ax.text(0.5, -0.22,
        f"Pearson r = {pearson_r:.3f}   |   Spearman ρ = {spearman_r:.3f}   |   "
        f"Median |Δ| = {median_abs_change:.3f}   |   Same direction = {same_direction:.1f}%",
        transform=ax.transAxes, ha="center", va="top", fontsize=8, color="0.3",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f7", ec="0.8", alpha=0.95))

plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
plt.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
print(f"Figure saved → {OUT_FIG}")
