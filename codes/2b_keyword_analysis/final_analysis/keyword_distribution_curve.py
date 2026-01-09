#!/usr/bin/env python3
"""
Keyword Distribution Curve Visualization

Shows the full distribution of all keywords by tweet count,
with the region containing target keywords highlighted.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

TARGET_KEYWORDS = [
    'aatmanirbhar', 'ayodhya', 'balochistan', 'bhakts',
    'democracy', 'demonetisation', 'dictatorship', 'gdp',
    'hathras', 'inflation', 'islamists', 'lynching',
    'mahotsav', 'minorities', 'msp', 'ratetvdebate',
    'sangh', 'sharia', 'spyware', 'suicides',
    'ucc', 'unemployment'
]

def main():
    # Load data
    csv_path = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/2b_keyword_analysis/keyword_by_stance.csv"
    output_dir = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/2b_keyword_analysis/final_analysis"
    
    df = pd.read_csv(csv_path)
    df_sorted = df.sort_values('Total', ascending=False).reset_index(drop=True)
    df_sorted['Rank'] = df_sorted.index + 1
    
    # Find target keyword ranks
    target_ranks = []
    target_totals = []
    for keyword in TARGET_KEYWORDS:
        matches = df_sorted[df_sorted['keyword'].str.lower() == keyword.lower()]
        if len(matches) > 0:
            target_ranks.append(matches.iloc[0]['Rank'])
            target_totals.append(matches.iloc[0]['Total'])
    
    min_rank, max_rank = min(target_ranks), max(target_ranks)
    min_tweets, max_tweets = min(target_totals), max(target_totals)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot full distribution curve
    ax.fill_between(df_sorted['Rank'], df_sorted['Total'], alpha=0.3, color='#2C3E50', label='All Keywords')
    ax.plot(df_sorted['Rank'], df_sorted['Total'], color='#2C3E50', linewidth=1.5)
    
    # Highlight the region where target keywords fall
    mask = (df_sorted['Rank'] >= min_rank) & (df_sorted['Rank'] <= max_rank)
    ax.fill_between(df_sorted.loc[mask, 'Rank'], df_sorted.loc[mask, 'Total'], 
                    alpha=0.6, color='#E74C3C', label=f'Target Keywords Region (Rank {min_rank} to {max_rank})')
    
    # Add vertical lines for min/max rank boundaries
    ax.axvline(x=min_rank, color='#E74C3C', linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(x=max_rank, color='#E74C3C', linestyle='--', linewidth=2, alpha=0.8)
    
    # Annotations
    ax.annotate(f'Highest: democracy\n(Rank #{min_rank}, {max_tweets:,} tweets)', 
                xy=(min_rank, max_tweets), xytext=(min_rank + 3000, max_tweets * 0.9),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#E74C3C'))
    
    ax.annotate(f'Lowest: ucc\n(Rank #{max_rank}, {min_tweets:,} tweets)', 
                xy=(max_rank, min_tweets), xytext=(max_rank + 5000, min_tweets + 5000),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#E74C3C'))
    
    # Add reference info
    total_keywords = len(df_sorted)
    coverage_pct = (1 - max_rank / total_keywords) * 100
    
    ax.text(0.98, 0.95, f'Total Keywords: {total_keywords:,}\n'
                        f'Target Keywords: {len(target_ranks)}\n'
                        f'All targets in Top {100-coverage_pct:.1f}%',
            transform=ax.transAxes, fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#ECF0F1', edgecolor='#BDC3C7'))
    
    ax.set_xlabel('Keyword Rank (by Total Tweet Count)', fontsize=12)
    ax.set_ylabel('Total Tweet Count', fontsize=12)
    ax.set_title('Full Keyword Distribution with Target Keywords Region Highlighted', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    
    ax.set_xlim(0, total_keywords)
    ax.set_ylim(0, df_sorted['Total'].max() * 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'keyword_distribution_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: keyword_distribution_curve.png")
    
    # Also create a zoomed version focusing on top keywords
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Show only top 10,000 keywords
    top_n = 10000
    df_top = df_sorted.head(top_n)
    
    ax.fill_between(df_top['Rank'], df_top['Total'], alpha=0.3, color='#2C3E50', label='All Keywords (Top 10K)')
    ax.plot(df_top['Rank'], df_top['Total'], color='#2C3E50', linewidth=1.5)
    
    # Highlight target region
    mask = (df_top['Rank'] >= min_rank) & (df_top['Rank'] <= max_rank)
    ax.fill_between(df_top.loc[mask, 'Rank'], df_top.loc[mask, 'Total'], 
                    alpha=0.6, color='#E74C3C', label=f'22 Target Keywords (Rank {min_rank}-{max_rank})')
    
    ax.axvline(x=min_rank, color='#E74C3C', linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(x=max_rank, color='#E74C3C', linestyle='--', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Keyword Rank (by Total Tweet Count)', fontsize=12)
    ax.set_ylabel('Total Tweet Count', fontsize=12)
    ax.set_title('Keyword Distribution (Top 10,000) with Target Keywords Highlighted', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'keyword_distribution_curve_zoomed.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: keyword_distribution_curve_zoomed.png")

if __name__ == "__main__":
    main()
