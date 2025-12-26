"""
Keyword Analysis Script
Analyzes keyword frequency and distribution by political stance (pro ruling vs pro opposition)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style for visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Paths
INPUT_FILE = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/final_data/tweets_exploded_by_keyword.csv"
OUTPUT_DIR = Path("/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/2b_keyword_analysis")

print("Loading data...")
df = pd.read_csv(INPUT_FILE)
print(f"Total rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")
print(f"\nUnique keywords: {df['keyword'].nunique()}")

# =============================================================================
# 1. Overall Keyword Frequency (Most Used to Least)
# =============================================================================
print("\n" + "="*60)
print("1. OVERALL KEYWORD FREQUENCY")
print("="*60)

keyword_counts = df['keyword'].value_counts()
print(f"\nTop 30 Keywords (Most to Least Used):")
print(keyword_counts.head(30))

# Save complete keyword list to file
keyword_list_df = pd.DataFrame({
    'keyword': keyword_counts.index,
    'total_count': keyword_counts.values,
    'percentage': (keyword_counts.values / keyword_counts.sum() * 100).round(4)
})
keyword_list_df.to_csv(OUTPUT_DIR / "keyword_frequency_list.csv", index=False)
print(f"\nSaved complete keyword list to: keyword_frequency_list.csv")

# =============================================================================
# 2. Keyword by Political Stance (Pro Ruling vs Pro Opposition)
# =============================================================================
print("\n" + "="*60)
print("2. KEYWORD FREQUENCY BY POLITICAL STANCE")
print("="*60)

# Check what stance-related columns exist
print(f"\nUnique 'side' values: {df['side'].unique()}")
print(f"Unique 'tweet_label' values: {df['tweet_label'].unique()}")

# Cross-tabulation of keywords by stance
keyword_by_stance = df.groupby(['keyword', 'tweet_label']).size().unstack(fill_value=0)
print(f"\nKeyword by stance shape: {keyword_by_stance.shape}")

# Add total and sort
keyword_by_stance['Total'] = keyword_by_stance.sum(axis=1)
keyword_by_stance = keyword_by_stance.sort_values('Total', ascending=False)

# Detect the actual column names for ruling and opposition
ruling_col = 'Pro Ruling' if 'Pro Ruling' in keyword_by_stance.columns else None
opp_col = 'Pro OPP' if 'Pro OPP' in keyword_by_stance.columns else ('Pro Opposition' if 'Pro Opposition' in keyword_by_stance.columns else None)

# Calculate percentages for each keyword
if ruling_col and opp_col:
    keyword_by_stance['Pro_Ruling_Pct'] = (keyword_by_stance[ruling_col] / keyword_by_stance['Total'] * 100).round(2)
    keyword_by_stance['Pro_Opposition_Pct'] = (keyword_by_stance[opp_col] / keyword_by_stance['Total'] * 100).round(2)
    print(f"\nUsing columns: Ruling='{ruling_col}', Opposition='{opp_col}'")

print("\nTop 30 Keywords by Total (with stance breakdown):")
print(keyword_by_stance.head(30))

# Save to CSV
keyword_by_stance.to_csv(OUTPUT_DIR / "keyword_by_stance.csv")
print(f"\nSaved keyword by stance analysis to: keyword_by_stance.csv")

# =============================================================================
# 3. Most Used Keywords by PRO RULING
# =============================================================================
print("\n" + "="*60)
print("3. TOP KEYWORDS BY PRO RULING")
print("="*60)

pro_ruling_df = df[df['tweet_label'] == 'Pro Ruling']
pro_ruling_keywords = pro_ruling_df['keyword'].value_counts()
print("\nTop 30 Keywords Used by Pro Ruling:")
print(pro_ruling_keywords.head(30))

# Save
pro_ruling_keywords.to_csv(OUTPUT_DIR / "keywords_pro_ruling.csv", header=['count'])

# =============================================================================
# 4. Most Used Keywords by PRO OPPOSITION (Pro OPP in this dataset)
# =============================================================================
print("\n" + "="*60)
print("4. TOP KEYWORDS BY PRO OPPOSITION")
print("="*60)

# Handle both 'Pro Opposition' and 'Pro OPP' as column names
opp_label = 'Pro OPP' if 'Pro OPP' in df['tweet_label'].unique() else 'Pro Opposition'
pro_opposition_df = df[df['tweet_label'] == opp_label]
pro_opposition_keywords = pro_opposition_df['keyword'].value_counts()
print(f"\nTop 30 Keywords Used by {opp_label}:")
print(pro_opposition_keywords.head(30))

# Save
pro_opposition_keywords.to_csv(OUTPUT_DIR / "keywords_pro_opposition.csv", header=['count'])

# =============================================================================
# 5. Keywords that are predominantly used by one side (Polar Keywords)
# =============================================================================
print("\n" + "="*60)
print("5. POLAR KEYWORDS ANALYSIS")
print("="*60)

# Keywords dominantly Pro Ruling (>70% Pro Ruling)
if 'Pro_Ruling_Pct' in keyword_by_stance.columns:
    polar_ruling = keyword_by_stance[keyword_by_stance['Pro_Ruling_Pct'] > 70][[ruling_col, opp_col, 'Total', 'Pro_Ruling_Pct']].sort_values('Total', ascending=False)
    print(f"\nKeywords Predominantly PRO RULING (>70%):")
    print(polar_ruling.head(20))

    # Keywords dominantly Pro Opposition (>70% Pro Opposition)
    polar_opposition = keyword_by_stance[keyword_by_stance['Pro_Opposition_Pct'] > 70][[ruling_col, opp_col, 'Total', 'Pro_Opposition_Pct']].sort_values('Total', ascending=False)
    print(f"\nKeywords Predominantly PRO OPPOSITION (>70%):")
    print(polar_opposition.head(20))

    # Keywords balanced (40-60% split)
    balanced = keyword_by_stance[(keyword_by_stance['Pro_Ruling_Pct'] >= 40) & (keyword_by_stance['Pro_Ruling_Pct'] <= 60)][[ruling_col, opp_col, 'Total', 'Pro_Ruling_Pct', 'Pro_Opposition_Pct']].sort_values('Total', ascending=False)
    print(f"\nBalanced Keywords (40-60% split):")
    print(balanced.head(20))

    # Save polar analysis
    polar_ruling.to_csv(OUTPUT_DIR / "polar_keywords_ruling.csv")
    polar_opposition.to_csv(OUTPUT_DIR / "polar_keywords_opposition.csv")
    balanced.to_csv(OUTPUT_DIR / "balanced_keywords.csv")
else:
    print("\nWarning: Unable to calculate polar keywords - percentage columns not available")
    polar_ruling = pd.DataFrame()
    polar_opposition = pd.DataFrame()

# =============================================================================
# 6. VISUALIZATIONS
# =============================================================================
print("\n" + "="*60)
print("6. GENERATING VISUALIZATIONS")
print("="*60)

# --- Visualization 1: Top 30 Keywords Overall ---
fig, ax = plt.subplots(figsize=(14, 10))
top_30 = keyword_counts.head(30)
colors = sns.color_palette("viridis", len(top_30))
bars = ax.barh(top_30.index[::-1], top_30.values[::-1], color=colors[::-1])
ax.set_xlabel('Number of Tweets', fontsize=12)
ax.set_ylabel('Keyword', fontsize=12)
ax.set_title('Top 30 Most Used Keywords', fontsize=14, fontweight='bold')
ax.bar_label(bars, fmt='%d', padding=3, fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "viz_top30_keywords_overall.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: viz_top30_keywords_overall.png")

# --- Visualization 2: Top 20 Keywords by Stance (Stacked) ---
fig, ax = plt.subplots(figsize=(14, 10))
stance_cols = [c for c in [ruling_col, opp_col] if c is not None]
top_20_stacked = keyword_by_stance.head(20)[stance_cols].iloc[::-1]
top_20_stacked.plot(kind='barh', stacked=True, ax=ax, color=['#2ecc71', '#e74c3c'], edgecolor='white')
ax.set_xlabel('Number of Tweets', fontsize=12)
ax.set_ylabel('Keyword', fontsize=12)
ax.set_title('Top 20 Keywords by Political Stance', fontsize=14, fontweight='bold')
ax.legend(title='Stance', loc='lower right')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "viz_top20_by_stance_stacked.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: viz_top20_by_stance_stacked.png")

# --- Visualization 3: Polar Keywords Comparison ---
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Pro Ruling dominant keywords
if 'polar_ruling' in dir() and len(polar_ruling) > 0:
    top_polar_ruling = polar_ruling.head(15)
    axes[0].barh(top_polar_ruling.index[::-1], top_polar_ruling['Total'].values[::-1], color='#2ecc71')
    axes[0].set_title('Top 15 Keywords\nDominantly Pro Ruling (>70%)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Total Tweets')
    for idx, (keyword, row) in enumerate(top_polar_ruling.iloc[::-1].iterrows()):
        axes[0].text(row['Total'] + 100, idx, f"{row['Pro_Ruling_Pct']:.0f}%", va='center', fontsize=9)

# Pro Opposition dominant keywords
if 'polar_opposition' in dir() and len(polar_opposition) > 0:
    top_polar_opposition = polar_opposition.head(15)
    axes[1].barh(top_polar_opposition.index[::-1], top_polar_opposition['Total'].values[::-1], color='#e74c3c')
    axes[1].set_title('Top 15 Keywords\nDominantly Pro Opposition (>70%)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Total Tweets')
    for idx, (keyword, row) in enumerate(top_polar_opposition.iloc[::-1].iterrows()):
        axes[1].text(row['Total'] + 100, idx, f"{row['Pro_Opposition_Pct']:.0f}%", va='center', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "viz_polar_keywords_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: viz_polar_keywords_comparison.png")

# --- Visualization 4: Top 30 Pro Ruling vs Pro Opposition Side by Side ---
fig, axes = plt.subplots(1, 2, figsize=(16, 10))

# Pro Ruling
top_30_ruling = pro_ruling_keywords.head(30)
axes[0].barh(top_30_ruling.index[::-1], top_30_ruling.values[::-1], color='#2ecc71', alpha=0.8)
axes[0].set_title('Top 30 Keywords: Pro Ruling', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Number of Tweets')

# Pro Opposition
top_30_opposition = pro_opposition_keywords.head(30)
axes[1].barh(top_30_opposition.index[::-1], top_30_opposition.values[::-1], color='#e74c3c', alpha=0.8)
axes[1].set_title('Top 30 Keywords: Pro Opposition', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Number of Tweets')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "viz_top30_ruling_vs_opposition.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: viz_top30_ruling_vs_opposition.png")

# --- Visualization 5: Stance Distribution Pie Chart ---
fig, ax = plt.subplots(figsize=(8, 8))
stance_totals = df['tweet_label'].value_counts()
colors = ['#2ecc71', '#e74c3c', '#95a5a6']
ax.pie(stance_totals.values, labels=stance_totals.index, autopct='%1.1f%%', colors=colors[:len(stance_totals)], startangle=90, explode=[0.02]*len(stance_totals))
ax.set_title('Overall Distribution of Tweet Stances', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "viz_stance_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: viz_stance_distribution.png")

# =============================================================================
# 7. FINAL KEYWORD LIST FOR STANCE ANALYSIS
# =============================================================================
print("\n" + "="*60)
print("7. FINAL KEYWORD LIST FOR STANCE ANALYSIS")
print("="*60)

unique_keywords = sorted(df['keyword'].dropna().unique().tolist())
print(f"\nTotal Unique Keywords: {len(unique_keywords)}")
print("\nFull Keyword List:")
for i, kw in enumerate(unique_keywords, 1):
    print(f"  {i}. {kw}")

# Save as JSON and TXT
with open(OUTPUT_DIR / "keywords_list.json", 'w') as f:
    json.dump(unique_keywords, f, indent=2)

with open(OUTPUT_DIR / "keywords_list.txt", 'w') as f:
    f.write('\n'.join(unique_keywords))

print(f"\nSaved keyword list to:")
print(f"  - keywords_list.json")
print(f"  - keywords_list.txt")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"\nOutput files generated in: {OUTPUT_DIR}")
print("""
Files Created:
  CSV Reports:
    - keyword_frequency_list.csv (all keywords ranked by frequency)
    - keyword_by_stance.csv (keywords with Pro Ruling/Opposition breakdown)
    - keywords_pro_ruling.csv (top keywords by Pro Ruling)
    - keywords_pro_opposition.csv (top keywords by Pro Opposition)
    - polar_keywords_ruling.csv (keywords >70% Pro Ruling)
    - polar_keywords_opposition.csv (keywords >70% Pro Opposition)
    - balanced_keywords.csv (keywords with 40-60% split)
  
  Keyword Lists:
    - keywords_list.json
    - keywords_list.txt
  
  Visualizations:
    - viz_top30_keywords_overall.png
    - viz_top20_by_stance_stacked.png
    - viz_polar_keywords_comparison.png
    - viz_top30_ruling_vs_opposition.png
    - viz_stance_distribution.png
""")
