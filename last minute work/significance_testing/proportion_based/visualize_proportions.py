#!/usr/bin/env python3
"""
Visualization for Proportion-Based Significance Testing
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

INPUT_DIR = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/last minute work/significance_testing/proportion_based"
OUTPUT_DIR = os.path.join(INPUT_DIR, "visualizations")


def load_results():
    """Load proportion-based test results."""
    intra_english = pd.read_csv(os.path.join(INPUT_DIR, "intra_english_proportions.csv"))
    intra_hindi = pd.read_csv(os.path.join(INPUT_DIR, "intra_hindi_proportions.csv"))
    inter_dataset = pd.read_csv(os.path.join(INPUT_DIR, "inter_dataset_proportions.csv"))
    return intra_english, intra_hindi, inter_dataset


def create_heatmap_with_diff(df, title, filename):
    """
    Create heatmap showing significance AND percentage difference.
    """
    # Pivot for significance
    pivot_sig = df.pivot_table(
        index='keyword', columns='stance', values='significant', aggfunc='first'
    ).fillna(False).astype(int)
    
    # Pivot for percentage difference
    pivot_diff = df.pivot_table(
        index='keyword', columns='stance', values='pct_difference', aggfunc='first'
    ).fillna(0)
    
    col_order = ['favor', 'against', 'neutral']
    pivot_sig = pivot_sig[[c for c in col_order if c in pivot_sig.columns]]
    pivot_diff = pivot_diff[[c for c in col_order if c in pivot_diff.columns]]
    
    # Sort by total significance
    pivot_sig['total'] = pivot_sig.sum(axis=1)
    sort_order = pivot_sig.sort_values('total', ascending=True).index
    pivot_sig = pivot_sig.loc[sort_order].drop('total', axis=1)
    pivot_diff = pivot_diff.loc[sort_order]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(8, len(pivot_sig) * 0.4)))
    
    # Use diverging colormap for percentage difference
    vmax = max(abs(pivot_diff.values.min()), abs(pivot_diff.values.max()))
    
    sns.heatmap(pivot_diff, cmap='RdBu_r', center=0, vmin=-vmax, vmax=vmax,
                ax=ax, linewidths=0.5, linecolor='white',
                annot=True, fmt='.0f', annot_kws={'size': 9, 'weight': 'bold'},
                cbar_kws={'label': '% Point Difference\n(Pro-Ruling - Pro-Opp)'})
    
    # Add significance markers
    for i, keyword in enumerate(pivot_sig.index):
        for j, stance in enumerate(pivot_sig.columns):
            if pivot_sig.loc[keyword, stance] == 1:
                ax.text(j + 0.9, i + 0.1, '*', fontsize=14, fontweight='bold', 
                       color='black', ha='center', va='center')
    
    ax.set_title(f'{title}\n(* = significant at p<0.05)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Stance', fontsize=11)
    ax.set_ylabel('Keyword', fontsize=11)
    ax.set_xticklabels([t.get_text().title() for t in ax.get_xticklabels()])
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filename}")


def create_summary_bar(intra_english, intra_hindi, inter_dataset):
    """Summary bar chart."""
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
    
    plt.suptitle('Proportion Z-Test: Significance Rates by Stance', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'summary_by_stance.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: summary_by_stance.png")


def create_top_differences_chart(df, title, filename, n_top=15):
    """
    Bar chart showing keywords with largest percentage differences.
    """
    # Get absolute differences for each stance
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    stance_labels = ['Favor', 'Against', 'Neutral']
    
    for ax, stance, color, label in zip(axes, ['favor', 'against', 'neutral'], colors, stance_labels):
        subset = df[(df['stance'] == stance) & (df['significant'] == True)].copy()
        subset['abs_diff'] = subset['pct_difference'].abs()
        subset = subset.nlargest(n_top, 'abs_diff').sort_values('pct_difference')
        
        if len(subset) == 0:
            ax.text(0.5, 0.5, 'No significant\ndifferences', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{label} Stance', fontsize=11, fontweight='bold')
            continue
        
        colors_bar = [color if v > 0 else '#7f8c8d' for v in subset['pct_difference']]
        bars = ax.barh(subset['keyword'], subset['pct_difference'], color=colors_bar, alpha=0.8)
        
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel('Percentage Point Difference\n(Pro-Ruling - Pro-Opposition)')
        ax.set_title(f'{label} Stance', fontsize=11, fontweight='bold')
    
    plt.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filename}")


def main():
    """Main execution."""
    print("=" * 60)
    print("PROPORTION-BASED VISUALIZATION")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\nLoading results...")
    intra_english, intra_hindi, inter_dataset = load_results()
    print(f"  English: {len(intra_english)} tests, {intra_english['significant'].sum()} significant")
    print(f"  Hindi: {len(intra_hindi)} tests, {intra_hindi['significant'].sum()} significant")
    print(f"  Inter-dataset: {len(inter_dataset)} tests, {inter_dataset['significant'].sum()} significant")
    
    print("\nCreating visualizations...")
    
    # Heatmaps with percentage differences
    create_heatmap_with_diff(intra_english, 'English: Percentage Point Difference by Keyword', 'heatmap_english.png')
    create_heatmap_with_diff(intra_hindi, 'Hindi: Percentage Point Difference by Keyword', 'heatmap_hindi.png')
    
    # Summary bar
    create_summary_bar(intra_english, intra_hindi, inter_dataset)
    
    # Top differences
    create_top_differences_chart(intra_english, 'English: Top Percentage Differences', 'top_differences_english.png')
    create_top_differences_chart(intra_hindi, 'Hindi: Top Percentage Differences', 'top_differences_hindi.png')
    
    print("\n" + "=" * 60)
    print(f"Visualizations saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
