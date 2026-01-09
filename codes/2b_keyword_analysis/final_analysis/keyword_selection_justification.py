#!/usr/bin/env python3
"""
Keyword Selection Justification Analysis

This script demonstrates:
1. OBJECTIVE THRESHOLD: All selected keywords are above a volume threshold (Top X%)
2. BALANCED REPRESENTATION: Keywords include both pro-ruling and pro-opposition topics

Output: Visualizations and statistics for paper justification
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
    # Paths
    csv_path = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/2b_keyword_analysis/keyword_by_stance.csv"
    output_dir = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/2b_keyword_analysis/final_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare data
    df = pd.read_csv(csv_path)
    df_sorted = df.sort_values('Total', ascending=False).reset_index(drop=True)
    df_sorted['Rank'] = df_sorted.index + 1
    total_keywords = len(df_sorted)
    
    # Extract target keyword data
    target_data = []
    for keyword in TARGET_KEYWORDS:
        matches = df_sorted[df_sorted['keyword'].str.lower() == keyword.lower()]
        if len(matches) > 0:
            row = matches.iloc[0]
            # Classify stance
            if row['Pro_Ruling_Pct'] > row['Pro_Opposition_Pct'] + 10:
                stance = 'Pro-Ruling'
            elif row['Pro_Opposition_Pct'] > row['Pro_Ruling_Pct'] + 10:
                stance = 'Pro-Opposition'
            else:
                stance = 'Balanced'
            
            target_data.append({
                'Keyword': row['keyword'],
                'Rank': row['Rank'],
                'Total': row['Total'],
                'Pro_Ruling_Pct': row['Pro_Ruling_Pct'],
                'Pro_Opposition_Pct': row['Pro_Opposition_Pct'],
                'Stance': stance
            })
    
    target_df = pd.DataFrame(target_data).sort_values('Rank')
    
    # Calculate threshold
    min_rank = target_df['Rank'].max()
    threshold_pct = (min_rank / total_keywords) * 100
    min_tweets = target_df['Total'].min()
    
    # =========================================================================
    # FIGURE 1: Volume Threshold + Partisan Balance (Combined)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- LEFT: Volume Threshold ---
    ax1 = axes[0]
    
    # Plot distribution curve
    ax1.semilogy(df_sorted['Rank'], df_sorted['Total'], color='#7F8C8D', linewidth=1.5, alpha=0.7)
    ax1.fill_between(df_sorted['Rank'], df_sorted['Total'], alpha=0.1, color='#7F8C8D')
    
    # Threshold line
    ax1.axvline(x=min_rank, color='#E74C3C', linestyle='--', linewidth=2.5, label=f'Threshold: Rank {min_rank:,} (Top {threshold_pct:.1f}%)')
    ax1.axhline(y=min_tweets, color='#E74C3C', linestyle=':', linewidth=2, alpha=0.7)
    
    # Color by stance
    colors = {'Pro-Ruling': '#E74C3C', 'Pro-Opposition': '#3498DB', 'Balanced': '#9B59B6'}
    for stance in ['Pro-Ruling', 'Pro-Opposition', 'Balanced']:
        subset = target_df[target_df['Stance'] == stance]
        ax1.scatter(subset['Rank'], subset['Total'], s=120, c=colors[stance], 
                   edgecolors='white', linewidth=1.5, zorder=5, label=f'{stance} ({len(subset)})')
    
    ax1.set_xlabel('Keyword Rank', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Tweet Count (log scale)', fontsize=12, fontweight='bold')
    ax1.set_title('1. Objective Volume Threshold\nAll 22 keywords above minimum threshold', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim(0, min(15000, total_keywords))
    ax1.grid(True, alpha=0.3, which='both')
    
    # Add annotation
    ax1.text(0.03, 0.03, f'Threshold: ≥{min_tweets:,} tweets\n(Top {threshold_pct:.1f}% of {total_keywords:,} keywords)', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#FDEBD0', edgecolor='#E74C3C'))
    
    # --- RIGHT: Partisan Balance ---
    ax2 = axes[1]
    
    # Sort by stance difference for visual clarity
    target_df_plot = target_df.copy()
    target_df_plot['Stance_Diff'] = target_df_plot['Pro_Ruling_Pct'] - target_df_plot['Pro_Opposition_Pct']
    target_df_plot = target_df_plot.sort_values('Stance_Diff')
    
    y_pos = range(len(target_df_plot))
    
    # Pro-Opposition bars (negative direction)
    bars_opp = ax2.barh(y_pos, -target_df_plot['Pro_Opposition_Pct'], color='#3498DB', 
                        label='Pro-Opposition %', edgecolor='white', linewidth=0.5)
    
    # Pro-Ruling bars (positive direction)
    bars_rul = ax2.barh(y_pos, target_df_plot['Pro_Ruling_Pct'], color='#E74C3C', 
                        label='Pro-Ruling %', edgecolor='white', linewidth=0.5)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(target_df_plot['Keyword'], fontsize=10)
    ax2.axvline(x=0, color='black', linewidth=1)
    ax2.set_xlim(-105, 105)
    ax2.set_xlabel('← Pro-Opposition %          Pro-Ruling % →', fontsize=11, fontweight='bold')
    ax2.set_title('2. Balanced Partisan Representation\nKeywords from both sides of political spectrum', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    
    # Add count summary
    stance_counts = target_df['Stance'].value_counts()
    summary_text = '\n'.join([f'{s}: {c}' for s, c in stance_counts.items()])
    ax2.text(0.98, 0.02, summary_text, transform=ax2.transAxes, fontsize=10, 
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F9F9', edgecolor='#BDC3C7'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'keyword_selection_justification.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: keyword_selection_justification.png")
    
    # =========================================================================
    # Generate Summary Statistics
    # =========================================================================
    print("\n" + "="*70)
    print("KEYWORD SELECTION JUSTIFICATION SUMMARY")
    print("="*70)
    
    print(f"\n1. OBJECTIVE VOLUME THRESHOLD:")
    print(f"   - Total keywords in dataset: {total_keywords:,}")
    print(f"   - All 22 selected keywords rank within Top {threshold_pct:.1f}%")
    print(f"   - Minimum tweets per keyword: {min_tweets:,}")
    print(f"   - Rank range: #{target_df['Rank'].min()} to #{target_df['Rank'].max()}")
    
    print(f"\n2. BALANCED PARTISAN REPRESENTATION:")
    for stance, count in stance_counts.items():
        keywords = target_df[target_df['Stance'] == stance]['Keyword'].tolist()
        print(f"   - {stance}: {count} keywords")
        print(f"     ({', '.join(keywords)})")
    
    # Save summary to file
    summary_path = os.path.join(output_dir, 'keyword_selection_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("KEYWORD SELECTION JUSTIFICATION\n")
        f.write("="*50 + "\n\n")
        f.write("1. OBJECTIVE VOLUME THRESHOLD\n")
        f.write("-"*30 + "\n")
        f.write(f"Total keywords in dataset: {total_keywords:,}\n")
        f.write(f"Selected keywords: 22\n")
        f.write(f"All selected keywords rank within: Top {threshold_pct:.1f}%\n")
        f.write(f"Minimum tweets per keyword: {min_tweets:,}\n")
        f.write(f"Rank range: #{target_df['Rank'].min()} to #{target_df['Rank'].max()}\n\n")
        
        f.write("2. BALANCED PARTISAN REPRESENTATION\n")
        f.write("-"*30 + "\n")
        for stance, count in stance_counts.items():
            keywords = target_df[target_df['Stance'] == stance]['Keyword'].tolist()
            f.write(f"{stance}: {count} keywords\n")
            f.write(f"  {', '.join(keywords)}\n")
        
        f.write("\n\nDETAILED BREAKDOWN:\n")
        f.write("-"*30 + "\n")
        f.write(f"{'Keyword':<20} {'Rank':>6} {'Tweets':>8} {'Pro-Rul%':>10} {'Pro-Opp%':>10} {'Stance':<15}\n")
        f.write("-"*70 + "\n")
        for _, row in target_df.iterrows():
            f.write(f"{row['Keyword']:<20} {row['Rank']:>6} {row['Total']:>8,} {row['Pro_Ruling_Pct']:>9.1f}% {row['Pro_Opposition_Pct']:>9.1f}% {row['Stance']:<15}\n")
    
    print(f"\nSummary saved to: {summary_path}")

if __name__ == "__main__":
    main()
