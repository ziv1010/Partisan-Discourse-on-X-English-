"""
Advanced Stance Visualization Extension
=======================================
This script adds more in-depth visualizations: scatterplots, overlap analysis, 
correlation heatmaps, and relationship diagrams.

Run this script in the same directory as stance_advanced_visualisation.ipynb
after running the main notebook.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Color schemes
STANCE_COLORS = {
    'favor': '#2ecc71',     # Green
    'against': '#e74c3c',   # Red
    'neutral': '#95a5a6'    # Gray
}

PARTY_COLORS = {
    'pro ruling': '#FF6B35',      # Orange
    'pro opposition': '#004E89'   # Blue
}

BUCKET_COLORS = {
    'Economic': '#1abc9c',
    'Social/Religious': '#9b59b6',
    'Political/Policy': '#3498db',
    'Leadership': '#e67e22'
}

# Load the summary data
if Path('advanced_stance_summary.csv').exists():
    polarization_df = pd.read_csv('advanced_stance_summary.csv')
    polarization_df.columns = ['keyword', 'bucket', 
                                'ruling_count', 'ruling_favor', 'ruling_against', 'ruling_neutral',
                                'opp_count', 'opp_favor', 'opp_against', 'opp_neutral',
                                'polarization_score', 'stance_divergence']
    print(f"Loaded {len(polarization_df)} keywords from summary")
else:
    print("ERROR: advanced_stance_summary.csv not found!")
    print("Please run the main notebook first.")
    exit(1)

# Also load the main data for detailed analysis
CSV1 = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/6_stance/results/stance_results_23keywords_v2.csv"
CSV2 = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/6_stance/results/stance_results_run1_15_keywords.csv"

dfs = []
for csv_path in [CSV1, CSV2]:
    if Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        dfs.append(df)

df = pd.concat(dfs, axis=0, ignore_index=True)
df = df.drop_duplicates(subset=['source_row', 'keyword'], keep='first')

# Filter and standardize
valid_stances = ['favor', 'against', 'neutral']
df = df[df['fewshot_label'].isin(valid_stances)].copy()
df['stance'] = df['fewshot_label']
df['party'] = df['_label_norm'].str.lower().str.strip()
df['keyword'] = df['keyword'].str.lower().str.strip()
df = df[df['party'].isin(['pro ruling', 'pro opposition'])]

print(f"Loaded {len(df):,} tweets for detailed analysis")

# ============================================================================
# 1. SCATTERPLOT: FAVOR vs AGAINST DIFFERENCE
# ============================================================================
print("\nGenerating: Stance Divergence Scatterplot...")

fig, ax = plt.subplots(figsize=(14, 10))

scatter_data = polarization_df.copy()
scatter_data['total'] = scatter_data['ruling_count'] + scatter_data['opp_count']
sizes = (scatter_data['total'] / scatter_data['total'].max()) * 500 + 50
colors = scatter_data['stance_divergence']

scatter_data['favor_diff'] = scatter_data['ruling_favor'] - scatter_data['opp_favor']
scatter_data['against_diff'] = scatter_data['ruling_against'] - scatter_data['opp_against']

scatter = ax.scatter(scatter_data['favor_diff'], scatter_data['against_diff'], 
                     s=sizes, c=colors, cmap='RdBu', alpha=0.7, edgecolors='black', linewidth=1)

for idx, row in scatter_data.head(10).iterrows():
    ax.annotate(row['keyword'], (row['favor_diff'], row['against_diff']),
                fontsize=9, fontweight='bold', ha='center', va='bottom',
                xytext=(0, 8), textcoords='offset points')

ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

ax.text(40, 50, 'Ruling: More Favor & Against', fontsize=10, ha='center', alpha=0.5)
ax.text(-40, 50, 'Ruling: Less Favor, More Against', fontsize=10, ha='center', alpha=0.5)
ax.text(40, -50, 'Ruling: More Favor, Less Against', fontsize=10, ha='center', alpha=0.5)
ax.text(-40, -50, 'Ruling: Less Favor & Against', fontsize=10, ha='center', alpha=0.5)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Stance Divergence Score', fontsize=12)

ax.set_xlabel('Favor Rate Difference (Ruling - Opposition) %', fontweight='bold', fontsize=12)
ax.set_ylabel('Against Rate Difference (Ruling - Opposition) %', fontweight='bold', fontsize=12)
ax.set_title('Stance Divergence Scatterplot: How Parties Differ on Keywords\n(Size = Tweet Volume, Color = Divergence)', 
            fontweight='bold', fontsize=14)

plt.tight_layout()
plt.savefig('stance_scatterplot_divergence.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: stance_scatterplot_divergence.png")

# ============================================================================
# 2. SCATTERPLOT: TWEET VOLUME BY PARTY
# ============================================================================
print("Generating: Keyword Volume Scatterplot...")

fig, ax = plt.subplots(figsize=(14, 10))

scatter_vol = polarization_df.copy()
colors_vol = [BUCKET_COLORS.get(b, '#7f8c8d') for b in scatter_vol['bucket']]

ax.scatter(scatter_vol['ruling_count'], scatter_vol['opp_count'], 
           s=100, c=colors_vol, alpha=0.7, edgecolors='black', linewidth=1)

for idx, row in scatter_vol.iterrows():
    ax.annotate(row['keyword'], (row['ruling_count'], row['opp_count']),
                fontsize=8, ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')

max_count = max(scatter_vol['ruling_count'].max(), scatter_vol['opp_count'].max())
ax.plot([0, max_count], [0, max_count], 'k--', alpha=0.3, label='Equal representation')

legend_elements = [Patch(facecolor=color, label=bucket) for bucket, color in BUCKET_COLORS.items()]
legend_elements.append(Patch(facecolor='#7f8c8d', label='Other'))
ax.legend(handles=legend_elements, loc='upper left', title='Topic Bucket')

ax.set_xlabel('Pro Ruling Tweet Count', fontweight='bold', fontsize=12)
ax.set_ylabel('Pro Opposition Tweet Count', fontweight='bold', fontsize=12)
ax.set_title('Keyword Volume Comparison: Which Party Tweets More About What?', fontweight='bold', fontsize=14)

plt.tight_layout()
plt.savefig('keyword_volume_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: keyword_volume_scatter.png")

# ============================================================================
# 3. OVERLAP ANALYSIS
# ============================================================================
print("Generating: Stance Overlap Analysis...")

def calculate_overlap_score(row):
    favor_overlap = 100 - abs(row['ruling_favor'] - row['opp_favor'])
    against_overlap = 100 - abs(row['ruling_against'] - row['opp_against'])
    neutral_overlap = 100 - abs(row['ruling_neutral'] - row['opp_neutral'])
    return (favor_overlap + against_overlap + neutral_overlap) / 3

overlap_df = polarization_df.copy()
overlap_df['overlap_score'] = overlap_df.apply(calculate_overlap_score, axis=1)
overlap_df = overlap_df.sort_values('overlap_score', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(18, 10))

high_overlap = overlap_df.head(10)
colors_high = [BUCKET_COLORS.get(b, '#7f8c8d') for b in high_overlap['bucket']]
axes[0].barh(high_overlap['keyword'], high_overlap['overlap_score'], 
             color=colors_high, edgecolor='white')
axes[0].set_xlabel('Overlap Score (%)', fontweight='bold')
axes[0].set_title('Keywords Where Parties AGREE Most\n(High Stance Overlap)', fontweight='bold', fontsize=13, color='green')
axes[0].invert_yaxis()
axes[0].set_xlim(60, 100)

low_overlap = overlap_df.tail(10).sort_values('overlap_score', ascending=True)
colors_low = [BUCKET_COLORS.get(b, '#7f8c8d') for b in low_overlap['bucket']]
axes[1].barh(low_overlap['keyword'], low_overlap['overlap_score'], 
             color=colors_low, edgecolor='white')
axes[1].set_xlabel('Overlap Score (%)', fontweight='bold')
axes[1].set_title('Keywords Where Parties DISAGREE Most\n(Low Stance Overlap)', fontweight='bold', fontsize=13, color='red')
axes[1].invert_yaxis()
axes[1].set_xlim(0, 80)

plt.suptitle('Stance Overlap Analysis: Where Do Parties Agree/Disagree?', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('stance_overlap_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: stance_overlap_analysis.png")

# ============================================================================
# 4. CORRELATION HEATMAP
# ============================================================================
print("Generating: Correlation Matrix...")

corr_cols = ['ruling_favor', 'ruling_against', 'ruling_neutral',
             'opp_favor', 'opp_against', 'opp_neutral',
             'polarization_score', 'stance_divergence']

corr_matrix = polarization_df[corr_cols].corr()

corr_matrix.columns = ['Ruling Favor', 'Ruling Against', 'Ruling Neutral',
                       'Opp Favor', 'Opp Against', 'Opp Neutral',
                       'Polarization', 'Divergence']
corr_matrix.index = corr_matrix.columns

fig, ax = plt.subplots(figsize=(12, 10))

mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, ax=ax, mask=mask, linewidths=0.5,
            cbar_kws={'label': 'Correlation', 'shrink': 0.8})

ax.set_title('Correlation Matrix: Stance Metrics Relationships', fontweight='bold', fontsize=14)

plt.tight_layout()
plt.savefig('stance_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: stance_correlation_matrix.png")

# ============================================================================
# 5. STANCE SHIFT DIAGRAM (SLOPE CHART)
# ============================================================================
print("Generating: Stance Shift Diagram...")

top_kws = polarization_df.nlargest(12, 'ruling_count')['keyword'].tolist()

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.flatten()

for idx, keyword in enumerate(top_kws):
    row = polarization_df[polarization_df['keyword'] == keyword].iloc[0]
    
    parties = ['Pro Ruling', 'Pro Opposition']
    favor_vals = [row['ruling_favor'], row['opp_favor']]
    against_vals = [row['ruling_against'], row['opp_against']]
    neutral_vals = [row['ruling_neutral'], row['opp_neutral']]
    
    x = [0, 1]
    
    axes[idx].plot(x, favor_vals, 'o-', color=STANCE_COLORS['favor'], linewidth=2, markersize=8, label='Favor')
    axes[idx].plot(x, against_vals, 'o-', color=STANCE_COLORS['against'], linewidth=2, markersize=8, label='Against')
    axes[idx].plot(x, neutral_vals, 'o-', color=STANCE_COLORS['neutral'], linewidth=2, markersize=8, label='Neutral')
    
    for i, party in enumerate(parties):
        axes[idx].annotate(f'{favor_vals[i]:.0f}%', (x[i], favor_vals[i]), fontsize=8, ha='center', va='bottom')
        axes[idx].annotate(f'{against_vals[i]:.0f}%', (x[i], against_vals[i]), fontsize=8, ha='center', va='bottom')
    
    axes[idx].set_xticks(x)
    axes[idx].set_xticklabels(parties, fontsize=9)
    axes[idx].set_ylabel('Stance %', fontsize=9)
    axes[idx].set_ylim(0, 100)
    
    title_color = BUCKET_COLORS.get(row['bucket'], '#333')
    axes[idx].set_title(keyword.upper(), fontweight='bold', fontsize=11, color=title_color)
    axes[idx].grid(axis='y', alpha=0.3)

axes[-1].legend(loc='upper right', fontsize=9)

plt.suptitle('Stance Shift Diagram: How Stance Changes Between Parties', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('stance_shift_diagram.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: stance_shift_diagram.png")

# ============================================================================
# 6. JOINT DISTRIBUTION HEATMAP
# ============================================================================
print("Generating: Joint Distribution Heatmap...")

joint_data = df.groupby(['keyword', 'party', 'stance']).size().reset_index(name='count')
joint_pivot = joint_data.pivot_table(index='keyword', columns=['party', 'stance'], values='count', fill_value=0)

top_keywords = df['keyword'].value_counts().head(15).index.tolist()
joint_pivot_filtered = joint_pivot.loc[top_keywords].astype(float)

joint_pivot_log = np.log1p(joint_pivot_filtered)

fig, ax = plt.subplots(figsize=(16, 12))
sns.heatmap(joint_pivot_log, annot=False, cmap='YlOrRd', ax=ax,
            cbar_kws={'label': 'log(Count + 1)'})

ax.set_title('Joint Distribution: Keyword × Party × Stance\n(Log scale for visibility)', 
            fontweight='bold', fontsize=14)
ax.set_ylabel('Keyword', fontweight='bold')
ax.set_xlabel('Party & Stance', fontweight='bold')

plt.tight_layout()
plt.savefig('joint_distribution_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: joint_distribution_heatmap.png")

# ============================================================================
# 7. RADAR CHART: PARTY FOCUS ACROSS BUCKETS
# ============================================================================
print("Generating: Party Focus Radar Chart...")

# Bucket classification function
KEYWORD_BUCKETS = {
    'Economic': ['gdp', 'inflation', 'unemployment', 'msp', 'aatmanirbhar', 'economy', 'budget'],
    'Social/Religious': ['ayodhya', 'minorities', 'islamists', 'sangh', 'hathras', 'ucc', 
                         'hindu', 'hindus', 'muslim', 'muslims', 'ram'],
    'Political/Policy': ['caa', 'farm', 'democracy', 'mahotsav', 'bjp', 'congress', 
                         'election', 'protest', 'farmers'],
    'Leadership': ['modi', 'bhakts', 'rahul', 'gandhi', 'pm']
}

def get_bucket(keyword):
    keyword_lower = keyword.lower()
    for bucket, keywords in KEYWORD_BUCKETS.items():
        for kw in keywords:
            if kw in keyword_lower:
                return bucket
    return 'Other'

df['bucket'] = df['keyword'].apply(get_bucket)

bucket_counts = df.groupby(['party', 'bucket']).size().unstack(fill_value=0)
bucket_pct = bucket_counts.div(bucket_counts.sum(axis=1), axis=0) * 100

categories = bucket_pct.columns.tolist()
N = len(categories)

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

for party, color in PARTY_COLORS.items():
    if party in bucket_pct.index:
        values = bucket_pct.loc[party].values.tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=party.title(), color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_title('Party Focus Across Topic Buckets\n(Radar Chart)', fontweight='bold', fontsize=14, y=1.08)
ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1))

plt.tight_layout()
plt.savefig('party_focus_radar.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: party_focus_radar.png")

# ============================================================================
# SUMMARY
# ============================================================================
import os

print("\n" + "=" * 80)
print("ADVANCED VISUALIZATIONS COMPLETE")
print("=" * 80)

viz_files = sorted([f for f in os.listdir('.') if f.endswith('.png')])
for i, f in enumerate(viz_files, 1):
    size = os.path.getsize(f) / 1024
    print(f"  {i}. {f} ({size:.1f} KB)")

print(f"\n📊 Total: {len(viz_files)} visualizations generated")
print("=" * 80)
