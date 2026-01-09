#!/usr/bin/env python3
"""
Keyword Ranking Visualization

Creates visual representations of where target keywords rank
compared to all keywords in the dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Define the target keywords
TARGET_KEYWORDS = [
    'aatmanirbhar', 'ayodhya', 'balochistan', 'bhakts',
    'democracy', 'demonetisation', 'dictatorship', 'gdp',
    'hathras', 'inflation', 'islamists', 'lynching',
    'mahotsav', 'minorities', 'msp', 'ratetvdebate',
    'sangh', 'sharia', 'spyware', 'suicides',
    'ucc', 'unemployment'
]

def load_and_prepare_data(csv_path):
    """Load data and calculate rankings."""
    df = pd.read_csv(csv_path)
    df_sorted = df.sort_values('Total', ascending=False).reset_index(drop=True)
    df_sorted['Rank'] = df_sorted.index + 1
    return df_sorted

def create_visualizations(df_sorted, output_dir):
    """Create multiple visualizations."""
    
    total_keywords = len(df_sorted)
    
    # Find target keywords
    target_data = []
    for keyword in TARGET_KEYWORDS:
        matches = df_sorted[df_sorted['keyword'].str.lower() == keyword.lower()]
        if len(matches) > 0:
            row = matches.iloc[0]
            target_data.append({
                'Keyword': row['keyword'],
                'Rank': row['Rank'],
                'Total': row['Total'],
                'Pro_Ruling_Pct': row['Pro_Ruling_Pct'],
                'Pro_Opposition_Pct': row['Pro_Opposition_Pct']
            })
    
    target_df = pd.DataFrame(target_data).sort_values('Rank')
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # =========================================================================
    # Figure 1: Rank Distribution Bar Chart
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(target_df)))
    
    bars = ax.barh(range(len(target_df)), target_df['Rank'], color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(range(len(target_df)))
    ax.set_yticklabels(target_df['Keyword'], fontsize=11)
    ax.set_xlabel('Rank (out of {:,} keywords)'.format(total_keywords), fontsize=12)
    ax.set_title('Target Keywords Ranking by Total Tweet Count', fontsize=14, fontweight='bold')
    
    # Add rank labels on bars
    for i, (rank, keyword) in enumerate(zip(target_df['Rank'], target_df['Keyword'])):
        ax.text(rank + 50, i, f'#{rank:,}', va='center', fontsize=9, fontweight='bold')
    
    ax.invert_yaxis()
    ax.set_xlim(0, max(target_df['Rank']) * 1.15)
    
    # Add reference lines
    percentile_ranks = {
        'Top 1%': int(total_keywords * 0.01),
        'Top 5%': int(total_keywords * 0.05),
        'Top 10%': int(total_keywords * 0.10)
    }
    
    for label, rank in percentile_ranks.items():
        if rank < max(target_df['Rank']) * 1.1:
            ax.axvline(x=rank, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
            ax.text(rank, len(target_df) + 0.3, label, ha='center', fontsize=9, color='red')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'keyword_rank_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: keyword_rank_distribution.png")
    
    # =========================================================================
    # Figure 2: Tweet Count Bar Chart with Log Scale
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Color by stance (red = pro-opposition, blue = pro-ruling)
    stance_colors = []
    for _, row in target_df.iterrows():
        if row['Pro_Ruling_Pct'] > row['Pro_Opposition_Pct']:
            stance_colors.append('#FF6B35')  # Orange for pro-ruling
        else:
            stance_colors.append('#4ECDC4')  # Teal for pro-opposition
    
    bars = ax.barh(range(len(target_df)), target_df['Total'], color=stance_colors, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(range(len(target_df)))
    ax.set_yticklabels(target_df['Keyword'], fontsize=11)
    ax.set_xlabel('Total Tweet Count', fontsize=12)
    ax.set_title('Target Keywords by Total Tweet Count\n(Orange = Pro-Ruling leaning, Teal = Pro-Opposition leaning)', fontsize=14, fontweight='bold')
    
    # Add count labels
    for i, total in enumerate(target_df['Total']):
        ax.text(total + 100, i, f'{total:,}', va='center', fontsize=9)
    
    ax.invert_yaxis()
    ax.set_xlim(0, max(target_df['Total']) * 1.15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'keyword_tweet_counts.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: keyword_tweet_counts.png")
    
    # =========================================================================
    # Figure 3: Scatter Plot - Rank vs Tweet Count with Stance Coloring
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create colormap based on stance ratio
    stance_ratio = target_df['Pro_Ruling_Pct'] - target_df['Pro_Opposition_Pct']
    
    scatter = ax.scatter(target_df['Rank'], target_df['Total'], 
                        c=stance_ratio, cmap='RdBu_r', 
                        s=150, edgecolors='black', linewidth=0.5,
                        vmin=-100, vmax=100)
    
    # Add labels
    for _, row in target_df.iterrows():
        ax.annotate(row['Keyword'], (row['Rank'], row['Total']), 
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Rank (out of {:,} keywords)'.format(total_keywords), fontsize=12)
    ax.set_ylabel('Total Tweet Count', fontsize=12)
    ax.set_title('Keyword Ranking vs Tweet Volume\n(Color: Blue = Pro-Opposition, Red = Pro-Ruling)', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Stance Difference (Pro_Ruling - Pro_Opposition)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'keyword_rank_vs_volume.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: keyword_rank_vs_volume.png")
    
    # =========================================================================
    # Figure 4: Log-scale Distribution showing position in overall dataset
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot all keywords as background
    ax.fill_between(range(1, total_keywords + 1), df_sorted['Total'], alpha=0.3, color='gray', label='All keywords')
    
    # Highlight target keywords
    for _, row in target_df.iterrows():
        ax.scatter(row['Rank'], row['Total'], s=100, zorder=5, edgecolors='black')
        ax.annotate(row['Keyword'], (row['Rank'], row['Total']), 
                   xytext=(3, 3), textcoords='offset points', fontsize=7, rotation=45)
    
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Rank (log scale)', fontsize=12)
    ax.set_ylabel('Total Tweet Count (log scale)', fontsize=12)
    ax.set_title('Target Keywords Position in Overall Keyword Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'keyword_distribution_overview.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: keyword_distribution_overview.png")
    
    print(f"\nAll visualizations saved to: {output_dir}")

def main():
    input_csv = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/2b_keyword_analysis/keyword_by_stance.csv"
    output_dir = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/2b_keyword_analysis/final_analysis"
    
    os.makedirs(output_dir, exist_ok=True)
    
    df_sorted = load_and_prepare_data(input_csv)
    create_visualizations(df_sorted, output_dir)

if __name__ == "__main__":
    main()
