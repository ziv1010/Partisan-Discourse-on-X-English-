#!/usr/bin/env python3
"""
Improved Keyword Distribution Visualization
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 15 seed aspects identified manually
SEED_ASPECTS = [
    'caa', 'congress', 'farm_laws', 'farmers_protests',
    'hindu', 'hindutva', 'kashmir', 'kashmiri_pandits',
    'modi', 'muslim', 'new_parliament', 'rahulgandhi',
    'ram_mandir', 'shaheen_bagh', 'china'
]

# 20 extended aspects identified using frequency analysis
EXTENDED_ASPECTS = [
    'aatmanirbhar', 'ayodhya', 'balochistan', 'bhakts',
    'democracy', 'demonetisation', 'dictatorship', 'gdp',
    'hathras', 'inflation', 'islamists', 'lynching',
    'mahotsav', 'minorities', 'msp', 'unemployment',
    'sangh', 'sharia', 'spyware', 'suicides'
]

# Combined list of all 35 aspects
TARGET_KEYWORDS = SEED_ASPECTS + EXTENDED_ASPECTS

def main():
    csv_path = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/2b_keyword_analysis/keyword_by_stance.csv"
    output_dir = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/2b_keyword_analysis/final_analysis"
    
    df = pd.read_csv(csv_path)
    df_sorted = df.sort_values('Total', ascending=False).reset_index(drop=True)
    df_sorted['Rank'] = df_sorted.index + 1
    
    # Find target keywords
    target_ranks = []
    target_totals = []
    target_names = []
    for keyword in TARGET_KEYWORDS:
        matches = df_sorted[df_sorted['keyword'].str.lower() == keyword.lower()]
        if len(matches) > 0:
            target_ranks.append(matches.iloc[0]['Rank'])
            target_totals.append(matches.iloc[0]['Total'])
            target_names.append(matches.iloc[0]['keyword'])
    
    min_rank, max_rank = min(target_ranks), max(target_ranks)
    total_keywords = len(df_sorted)
    
    # =========================================================================
    # IMPROVED PLOT: Log-scale with clear region marking
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot full distribution as a line (log scale on Y)
    ax.semilogy(df_sorted['Rank'], df_sorted['Total'], color='#34495E', linewidth=2, label='All Aspects Distribution')
    
    # Fill under the curve
    ax.fill_between(df_sorted['Rank'], df_sorted['Total'], alpha=0.15, color='#34495E')
    
    # Highlight target keywords region with shaded band
    ax.axvspan(min_rank, max_rank, alpha=0.3, color='#E74C3C', label=f'Target Aspects Zone\n(Rank {min_rank} to {max_rank})')
    
    # Plot individual target keywords as points
    ax.scatter(target_ranks, target_totals, s=80, c='#E74C3C', edgecolors='white', linewidth=1.5, zorder=5, label='35 Target Aspects')
    
    # Add key annotations
    # Highest ranked target
    highest_idx = target_totals.index(max(target_totals))
    ax.annotate(f'{target_names[highest_idx]}\n#{target_ranks[highest_idx]}', 
                xy=(target_ranks[highest_idx], target_totals[highest_idx]),
                xytext=(target_ranks[highest_idx] + 2000, target_totals[highest_idx] * 1.5),
                fontsize=9, fontweight='bold', color='#C0392B',
                arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.5))
    
    # Lowest ranked target
    lowest_idx = target_totals.index(min(target_totals))
    ax.annotate(f'{target_names[lowest_idx]}\n#{target_ranks[lowest_idx]}', 
                xy=(target_ranks[lowest_idx], target_totals[lowest_idx]),
                xytext=(target_ranks[lowest_idx] + 3000, target_totals[lowest_idx] * 2),
                fontsize=9, fontweight='bold', color='#C0392B',
                arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.5))
    
    # Add reference lines at percentile thresholds
    for pct, style in [(1, '-'), (5, '--'), (10, ':')]:
        rank_thresh = int(total_keywords * pct / 100)
        ax.axvline(x=rank_thresh, color='#27AE60', linestyle=style, linewidth=1.5, alpha=0.7)
        ax.text(rank_thresh + 200, ax.get_ylim()[1] * 0.5, f'Top {pct}%', 
                rotation=90, va='center', fontsize=9, color='#27AE60', fontweight='bold')
    
    # Info box
    info_text = (f"Total Aspects: {total_keywords:,}\n"
                 f"Target Aspects: {len(target_ranks)}\n"
                 f"Range: Rank #{min_rank} to #{max_rank}\n"
                 f"All in Top 5%")
    ax.text(0.98, 0.98, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#BDC3C7', alpha=0.9))
    
    ax.set_xlabel('Aspect Rank (sorted by Total Tweet Count)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Tweet Count (log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Aspect Distribution: Where Do the 35 Target Aspects Fall?', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, bbox_to_anchor=(0.85, 0.85))
    ax.set_xlim(0, total_keywords)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'keyword_distribution_improved.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: keyword_distribution_improved.png")
    
    # =========================================================================
    # ZOOMED VERSION: Focus on top 10K keywords
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    
    top_n = 10000
    df_top = df_sorted.head(top_n)
    
    ax.semilogy(df_top['Rank'], df_top['Total'], color='#34495E', linewidth=2.5)
    ax.fill_between(df_top['Rank'], df_top['Total'], alpha=0.15, color='#34495E')
    
    # Highlight zone
    ax.axvspan(min_rank, min(max_rank, top_n), alpha=0.35, color='#E74C3C', 
               label=f'Target Aspects Zone (Rank {min_rank}-{max_rank})')
    
    # Plot targets
    visible_targets = [(r, t, n) for r, t, n in zip(target_ranks, target_totals, target_names) if r <= top_n]
    if visible_targets:
        ranks, totals, names = zip(*visible_targets)
        ax.scatter(ranks, totals, s=120, c='#E74C3C', edgecolors='white', linewidth=2, zorder=5)
        
        # Label some key points
        for r, t, n in visible_targets[:5]:  # Label top 5
            ax.annotate(n, xy=(r, t), xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, fontweight='bold', color='#C0392B')
    
    ax.set_xlabel('Aspect Rank', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Tweet Count (log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Top 10,000 Aspects Distribution with Target Aspects Highlighted', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim(0, top_n)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'keyword_distribution_zoomed_improved.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: keyword_distribution_zoomed_improved.png")

if __name__ == "__main__":
    main()
