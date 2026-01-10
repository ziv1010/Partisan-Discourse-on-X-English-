#!/usr/bin/env python3
"""
Visualization for Logistic Regression Significance Testing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = 10

INPUT_DIR = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/last minute work/significance_testing"
OUTPUT_DIR = os.path.join(INPUT_DIR, "visualizations")

# Source data paths
ENGLISH_DATA = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/final_results+visualisations_folder/stance_results_37keywords.csv"
HINDI_DATA = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/final_results+visualisations_folder/hindi_stance_results.csv"


def load_results():
    """Load logistic regression test results."""
    intra_english = pd.read_csv(os.path.join(INPUT_DIR, "intra_english_results.csv"))
    intra_hindi = pd.read_csv(os.path.join(INPUT_DIR, "intra_hindi_results.csv"))
    inter_dataset = pd.read_csv(os.path.join(INPUT_DIR, "inter_dataset_results.csv"))
    return intra_english, intra_hindi, inter_dataset


def load_source_data():
    """Load source tweet data for count analysis."""
    df_english = pd.read_csv(ENGLISH_DATA)
    df_hindi = pd.read_csv(HINDI_DATA)
    return df_english, df_hindi


def create_odds_ratio_heatmap(df, title, filename):
    """
    Create heatmap showing odds ratios by keyword and stance.
    Green = more for pro-ruling/English, Red = more for pro-opp/Hindi
    """
    # Pivot: keywords (rows) x stance (columns) with odds ratio
    pivot_or = df.pivot_table(
        index='keyword', columns='stance', values='odds_ratio', aggfunc='first'
    ).fillna(1)
    
    pivot_sig = df.pivot_table(
        index='keyword', columns='stance', values='significant', aggfunc='first'
    ).fillna(False)
    
    col_order = ['favor', 'against', 'neutral']
    pivot_or = pivot_or[[c for c in col_order if c in pivot_or.columns]]
    pivot_sig = pivot_sig[[c for c in col_order if c in pivot_sig.columns]]
    
    # Convert to log odds ratio for symmetric visualization
    log_or = np.log2(pivot_or.values + 0.01)  # Add small value to avoid log(0)
    
    # Sort by favor odds ratio
    sort_idx = np.argsort(pivot_or['favor'].values)
    pivot_or = pivot_or.iloc[sort_idx]
    pivot_sig = pivot_sig.iloc[sort_idx]
    log_or = log_or[sort_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(10, len(pivot_or) * 0.4)))
    
    # Use diverging colormap
    vmax = max(abs(log_or.min()), abs(log_or.max()))
    
    sns.heatmap(log_or, cmap='RdBu', center=0, vmin=-vmax, vmax=vmax,
                ax=ax, linewidths=0.5, linecolor='white',
                annot=pivot_or.values, fmt='.1f', annot_kws={'size': 9, 'weight': 'bold'},
                cbar_kws={'label': 'Log2(Odds Ratio)'})
    
    # Add significance markers
    for i in range(len(pivot_sig)):
        for j in range(len(pivot_sig.columns)):
            if pivot_sig.iloc[i, j]:
                ax.text(j + 0.9, i + 0.1, '*', fontsize=12, fontweight='bold', 
                       color='black', ha='center', va='center')
    
    ax.set_yticklabels(pivot_or.index, rotation=0)
    ax.set_xticklabels([t.capitalize() for t in col_order], rotation=0)
    
    ax.set_title(f'{title}\n(* = significant p<0.05, Blue = Pro-Ruling higher, Red = Pro-Opp higher)', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Stance', fontsize=11)
    ax.set_ylabel('Keyword', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filename}")


def create_forest_plot(df, title, filename, n_top=20):
    """
    Create forest plot showing odds ratios with confidence-style visualization.
    """
    # Get significant results sorted by odds ratio
    sig_df = df[df['significant'] == True].copy()
    
    if len(sig_df) == 0:
        print(f"  Skipping {filename}: No significant results")
        return
    
    # Calculate log odds ratio
    sig_df['log_or'] = np.log2(sig_df['odds_ratio'])
    sig_df['abs_log_or'] = abs(sig_df['log_or'])
    
    # Get top by absolute effect size
    sig_df = sig_df.nlargest(n_top, 'abs_log_or').sort_values('log_or')
    
    fig, ax = plt.subplots(figsize=(10, max(8, len(sig_df) * 0.4)))
    
    y_pos = np.arange(len(sig_df))
    colors = ['#e74c3c' if x < 0 else '#3498db' for x in sig_df['log_or']]
    
    ax.barh(y_pos, sig_df['log_or'], color=colors, alpha=0.8, edgecolor='white')
    ax.axvline(x=0, color='black', linewidth=1)
    
    ax.set_yticks(y_pos)
    labels = [f"{row['keyword']} ({row['stance']})" for _, row in sig_df.iterrows()]
    ax.set_yticklabels(labels)
    
    ax.set_xlabel('Log2(Odds Ratio)\n← Pro-Opposition more likely | Pro-Ruling more likely →', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Add OR values as text
    for i, (_, row) in enumerate(sig_df.iterrows()):
        x = row['log_or']
        ax.text(x + 0.1 if x >= 0 else x - 0.1, i, f"OR={row['odds_ratio']:.1f}", 
               va='center', ha='left' if x >= 0 else 'right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filename}")


def create_summary_bar(intra_english, intra_hindi, inter_dataset):
    """Summary bar chart showing significance rates by stance."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    datasets = [
        (intra_english, 'English\n(Pro-Ruling vs Pro-Opposition)', '#3498db'),
        (intra_hindi, 'Hindi\n(Pro-Ruling vs Pro-Opposition)', '#e74c3c'),
        (inter_dataset, 'Inter-Dataset\n(English vs Hindi)', '#9b59b6')
    ]
    
    for ax, (df, title, color) in zip(axes, datasets):
        by_stance = df.groupby('stance').agg({
            'significant': ['sum', 'count']
        }).reset_index()
        by_stance.columns = ['stance', 'significant', 'total']
        by_stance['rate'] = by_stance['significant'] / by_stance['total'] * 100
        by_stance['stance'] = by_stance['stance'].str.title()
        
        order = {'Favor': 0, 'Against': 1, 'Neutral': 2}
        by_stance['order'] = by_stance['stance'].map(order)
        by_stance = by_stance.sort_values('order')
        
        bars = ax.bar(by_stance['stance'], by_stance['rate'], color=color, alpha=0.8)
        
        for bar, sig, total in zip(bars, by_stance['significant'], by_stance['total']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                   f'{int(sig)}/{int(total)}', ha='center', fontsize=10, fontweight='bold')
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel('% Significant (p<0.05)')
        ax.set_ylim(0, 110)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    
    plt.suptitle('Logistic Regression: Significance Rates by Stance', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'summary_by_stance.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: summary_by_stance.png")


def create_tweet_count_heatmap(df, title, filename, language):
    """Create heatmap showing tweet counts per keyword/stance/political leaning."""
    stance_col = 'fewshot_label_for_against'
    keywords = df['keyword'].unique()
    
    data = []
    for keyword in keywords:
        row = {'keyword': keyword}
        for leaning in ['pro ruling', 'pro opposition']:
            for stance in ['favor', 'against', 'neutral']:
                mask = (df['keyword'] == keyword) & (df['_label_norm'] == leaning) & (df[stance_col] == stance)
                count = len(df[mask])
                row[f'{leaning}_{stance}'] = count
        data.append(row)
    
    count_df = pd.DataFrame(data).set_index('keyword')
    count_df['total'] = count_df.sum(axis=1)
    count_df = count_df.sort_values('total', ascending=True).drop('total', axis=1)
    
    fig, ax = plt.subplots(figsize=(14, max(10, len(count_df) * 0.4)))
    log_data = np.log10(count_df.values + 1)
    
    sns.heatmap(log_data, cmap='YlOrRd', ax=ax, linewidths=0.3, linecolor='white',
                annot=count_df.values, fmt='.0f', annot_kws={'size': 8},
                cbar_kws={'label': 'Log10(Count + 1)'})
    
    ax.set_yticklabels(count_df.index, rotation=0)
    col_labels = ['Pro-Ruling\nFavor', 'Pro-Ruling\nAgainst', 'Pro-Ruling\nNeutral',
                  'Pro-Opp\nFavor', 'Pro-Opp\nAgainst', 'Pro-Opp\nNeutral']
    ax.set_xticklabels(col_labels, rotation=0)
    
    ax.set_title(f'{title}\n(Tweet Counts)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Keyword')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filename}")


def create_significance_per_keyword(df, title, filename):
    """
    Create a clear heatmap showing which keywords are significant (✓) vs not (✗) for each stance.
    """
    # Pivot: keywords (rows) x stance (columns)
    pivot_sig = df.pivot_table(
        index='keyword', columns='stance', values='significant', aggfunc='first'
    ).fillna(False).astype(int)
    
    col_order = ['favor', 'against', 'neutral']
    pivot_sig = pivot_sig[[c for c in col_order if c in pivot_sig.columns]]
    
    # Sort by total significant
    pivot_sig['total_sig'] = pivot_sig.sum(axis=1)
    pivot_sig = pivot_sig.sort_values('total_sig', ascending=True).drop('total_sig', axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(10, len(pivot_sig) * 0.4)))
    
    # Green for significant, light red for not
    colors = ['#ffcccc', '#2ecc71']
    cmap = LinearSegmentedColormap.from_list('sig', colors, N=2)
    
    # Create annotation with symbols
    annot_labels = np.where(pivot_sig.values == 1, '✓', '✗')
    
    sns.heatmap(pivot_sig, cmap=cmap, cbar=False, ax=ax, 
                linewidths=0.5, linecolor='white',
                annot=annot_labels, fmt='', annot_kws={'size': 14, 'weight': 'bold'})
    
    ax.set_yticklabels(pivot_sig.index, rotation=0)
    ax.set_xticklabels([c.capitalize() for c in col_order], rotation=0)
    
    ax.set_title(f'{title}\n(✓ = Significant p<0.05, ✗ = Not Significant)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Stance', fontsize=11)
    ax.set_ylabel('Keyword', fontsize=11)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ffcccc', edgecolor='gray', label='Not Significant (p≥0.05)'),
        Patch(facecolor='#2ecc71', edgecolor='gray', label='Significant (p<0.05)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    
    # Add counts at bottom
    sig_counts = pivot_sig.sum()
    for i, col in enumerate(pivot_sig.columns):
        ax.text(i + 0.5, -0.5, f'{int(sig_counts[col])}/{len(pivot_sig)}', 
               ha='center', va='top', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filename}")


def main():
    """Main execution."""
    print("=" * 60)
    print("LOGISTIC REGRESSION VISUALIZATION")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\nLoading results...")
    intra_english, intra_hindi, inter_dataset = load_results()
    print(f"  English: {len(intra_english)} tests, {intra_english['significant'].sum()} significant")
    print(f"  Hindi: {len(intra_hindi)} tests, {intra_hindi['significant'].sum()} significant")
    print(f"  Inter-dataset: {len(inter_dataset)} tests, {inter_dataset['significant'].sum()} significant")
    
    print("\nLoading source data for tweet counts...")
    df_english, df_hindi = load_source_data()
    print(f"  English: {len(df_english):,} tweets")
    print(f"  Hindi: {len(df_hindi):,} tweets")
    
    print("\nCreating visualizations...")
    
    # Significance per keyword heatmaps (NEW - clear view of which passed)
    create_significance_per_keyword(intra_english, 'English: Significant Keywords per Stance', 'significance_english.png')
    create_significance_per_keyword(intra_hindi, 'Hindi: Significant Keywords per Stance', 'significance_hindi.png')
    
    # Inter-dataset significance (by political leaning)
    for leaning in ['pro ruling', 'pro opposition']:
        subset = inter_dataset[inter_dataset['political_leaning'] == leaning]
        leaning_label = leaning.replace(' ', '_')
        create_significance_per_keyword(subset, f'Inter-Dataset ({leaning.title()}): English vs Hindi', f'significance_inter_{leaning_label}.png')
    
    # Odds ratio heatmaps
    create_odds_ratio_heatmap(intra_english, 'English: Odds Ratios by Keyword & Stance', 'heatmap_english.png')
    create_odds_ratio_heatmap(intra_hindi, 'Hindi: Odds Ratios by Keyword & Stance', 'heatmap_hindi.png')
    
    # Forest plots
    create_forest_plot(intra_english, 'English: Top Significant Effects (Forest Plot)', 'forest_plot_english.png')
    create_forest_plot(intra_hindi, 'Hindi: Top Significant Effects (Forest Plot)', 'forest_plot_hindi.png')
    
    # Summary bar
    create_summary_bar(intra_english, intra_hindi, inter_dataset)
    
    # Tweet count heatmaps
    print("\nCreating tweet count visualizations...")
    create_tweet_count_heatmap(df_english, 'English Dataset', 'tweet_counts_english.png', 'English')
    create_tweet_count_heatmap(df_hindi, 'Hindi Dataset', 'tweet_counts_hindi.png', 'Hindi')
    
    print("\n" + "=" * 60)
    print(f"All visualizations saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

