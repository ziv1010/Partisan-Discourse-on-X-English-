#!/usr/bin/env python3
"""
Visualization Script for Binomial GLM Significance Testing Results

Creates multiple visualizations:
1. Forest plot - Odds ratios with confidence intervals
2. Heatmap - Significance and direction by keyword
3. Bar chart - Beta coefficients with significance indicators
4. Bubble chart - Tweet volume vs. odds ratio with significance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
import os

# File paths
RESULTS_FILE = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/last minute work/significance_testing/significance_results.csv"
OUTPUT_DIR = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/last minute work/significance_testing/visualizations"

# Modern color palette
COLORS = {
    'pro_ruling': '#E63946',      # Red for pro-ruling
    'opposition': '#1D3557',       # Dark blue for opposition
    'not_significant': '#ADB5BD',  # Gray for non-significant
    'background': '#FFFFFF',
    'grid': '#E9ECEF',
    'text': '#212529'
}


def load_results():
    """Load significance results."""
    df = pd.read_csv(RESULTS_FILE)
    print(f"Loaded {len(df)} keyword results")
    return df


def setup_plot_style():
    """Set up modern plot aesthetics."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.facecolor': COLORS['background'],
        'figure.facecolor': COLORS['background'],
    })


def plot_forest(df, output_path):
    """
    Create a forest plot showing odds ratios with confidence intervals.
    """
    # Sort by odds ratio
    df_sorted = df.sort_values('odds_ratio', ascending=True).copy()
    
    # Calculate confidence intervals (approximate using z=1.96)
    df_sorted['or_lower'] = np.exp(df_sorted['beta1'] - 1.96 * df_sorted['std_err'])
    df_sorted['or_upper'] = np.exp(df_sorted['beta1'] + 1.96 * df_sorted['std_err'])
    
    # Clip for visualization
    df_sorted['or_display'] = df_sorted['odds_ratio'].clip(0.01, 100)
    df_sorted['or_lower_display'] = df_sorted['or_lower'].clip(0.01, 100)
    df_sorted['or_upper_display'] = df_sorted['or_upper'].clip(0.01, 100)
    
    fig, ax = plt.subplots(figsize=(12, 14))
    
    y_positions = range(len(df_sorted))
    
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        if not row['significant']:
            color = COLORS['not_significant']
        elif row['beta1'] < 0:
            color = COLORS['pro_ruling']
        else:
            color = COLORS['opposition']
        
        # Error bars
        ax.errorbar(
            row['or_display'], i,
            xerr=[[row['or_display'] - row['or_lower_display']], 
                  [row['or_upper_display'] - row['or_display']]],
            fmt='o', color=color, markersize=8, capsize=4,
            elinewidth=2, capthick=1.5, alpha=0.8
        )
    
    # Reference line at odds ratio = 1
    ax.axvline(x=1, color='#6C757D', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax.set_xscale('log')
    ax.set_xlim(0.01, 100)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df_sorted['keyword'])
    ax.set_xlabel('Odds Ratio (log scale)', fontweight='bold', fontsize=12)
    ax.set_title('Forest Plot: Effect of Influencer Alignment on Favor Tweets\n(Odds Ratio with 95% CI)', 
                 fontweight='bold', fontsize=14, pad=20)
    
    # Add annotations
    ax.annotate('← Pro-ruling favors', xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=10, ha='left', va='top', color=COLORS['pro_ruling'], fontweight='bold')
    ax.annotate('Opposition favors →', xy=(0.98, 0.98), xycoords='axes fraction',
                fontsize=10, ha='right', va='top', color=COLORS['opposition'], fontweight='bold')
    
    # Legend
    legend_elements = [
        mpatches.Patch(color=COLORS['pro_ruling'], label='Pro-ruling more likely (p<0.05)'),
        mpatches.Patch(color=COLORS['opposition'], label='Opposition more likely (p<0.05)'),
        mpatches.Patch(color=COLORS['not_significant'], label='Not significant (p≥0.05)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {os.path.basename(output_path)}")


def plot_coefficient_bars(df, output_path):
    """
    Create a bar chart of beta coefficients with significance indicators.
    """
    # Sort by beta1
    df_sorted = df.sort_values('beta1', ascending=True).copy()
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = []
    for _, row in df_sorted.iterrows():
        if not row['significant']:
            colors.append(COLORS['not_significant'])
        elif row['beta1'] < 0:
            colors.append(COLORS['pro_ruling'])
        else:
            colors.append(COLORS['opposition'])
    
    bars = ax.barh(range(len(df_sorted)), df_sorted['beta1'], color=colors, 
                   edgecolor='white', linewidth=0.5)
    
    # Reference line at 0
    ax.axvline(x=0, color='#6C757D', linestyle='-', linewidth=1.5)
    
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['keyword'])
    ax.set_xlabel('β₁ Coefficient (log-odds scale)', fontweight='bold', fontsize=12)
    ax.set_title('Effect of Opposition Alignment on Favor Tweet Probability\n(β₁ > 0 = Opposition more likely to favor)', 
                 fontweight='bold', fontsize=14, pad=20)
    
    # Add significance stars
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        if row['significant']:
            star = '***' if row['p_value'] < 0.001 else ('**' if row['p_value'] < 0.01 else '*')
            x_pos = row['beta1'] + 0.1 if row['beta1'] >= 0 else row['beta1'] - 0.1
            ha = 'left' if row['beta1'] >= 0 else 'right'
            ax.annotate(star, (x_pos, i), ha=ha, va='center', fontsize=10, fontweight='bold')
    
    # Legend
    legend_elements = [
        mpatches.Patch(color=COLORS['pro_ruling'], label='Pro-ruling more likely'),
        mpatches.Patch(color=COLORS['opposition'], label='Opposition more likely'),
        mpatches.Patch(color=COLORS['not_significant'], label='Not significant'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
    
    # Add note
    ax.annotate('* p<0.05,  ** p<0.01,  *** p<0.001', xy=(0.02, 0.02), xycoords='axes fraction',
                fontsize=9, ha='left', style='italic', color='#6C757D')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {os.path.basename(output_path)}")


def plot_heatmap(df, output_path):
    """
    Create a heatmap showing direction and strength of effects.
    """
    # Prepare data for heatmap
    df_heat = df.copy()
    
    # Create effect strength metric (signed log odds ratio for visualization)
    df_heat['effect'] = df_heat['beta1']
    df_heat.loc[~df_heat['significant'], 'effect'] = 0  # Zero out non-significant
    
    # Sort by effect
    df_heat = df_heat.sort_values('effect', ascending=False)
    
    fig, ax = plt.subplots(figsize=(6, 12))
    
    # Create heatmap data
    effect_data = df_heat[['keyword', 'effect']].set_index('keyword')
    
    # Use diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    vmax = max(abs(effect_data['effect'].min()), abs(effect_data['effect'].max()))
    
    sns.heatmap(effect_data, annot=True, cmap=cmap, center=0, vmin=-vmax, vmax=vmax,
                fmt='.2f', linewidths=0.5, cbar_kws={'label': 'β₁ Coefficient'},
                ax=ax, annot_kws={'size': 9})
    
    ax.set_title('Effect Direction and Strength by Keyword\n(Red = Opposition favors, Blue = Pro-ruling favors)', 
                 fontweight='bold', fontsize=12, pad=15)
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {os.path.basename(output_path)}")


def plot_bubble_chart(df, output_path):
    """
    Create a bubble chart: tweet volume vs. odds ratio with bubble size = influencer count.
    """
    df_plot = df.copy()
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Separate significant and non-significant
    sig_pro_ruling = df_plot[(df_plot['significant']) & (df_plot['beta1'] < 0)]
    sig_opposition = df_plot[(df_plot['significant']) & (df_plot['beta1'] > 0)]
    not_sig = df_plot[~df_plot['significant']]
    
    # Size scaling
    size_scale = 0.15
    
    # Plot each group
    ax.scatter(sig_pro_ruling['total_tweets'], sig_pro_ruling['odds_ratio'],
               s=sig_pro_ruling['n_influencers'] * size_scale, alpha=0.6,
               c=COLORS['pro_ruling'], label='Pro-ruling (p<0.05)', edgecolors='white', linewidth=1)
    
    ax.scatter(sig_opposition['total_tweets'], sig_opposition['odds_ratio'],
               s=sig_opposition['n_influencers'] * size_scale, alpha=0.6,
               c=COLORS['opposition'], label='Opposition (p<0.05)', edgecolors='white', linewidth=1)
    
    ax.scatter(not_sig['total_tweets'], not_sig['odds_ratio'],
               s=not_sig['n_influencers'] * size_scale, alpha=0.4,
               c=COLORS['not_significant'], label='Not significant', edgecolors='white', linewidth=1)
    
    # Reference line
    ax.axhline(y=1, color='#6C757D', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Total Tweets (log scale)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Odds Ratio (log scale)', fontweight='bold', fontsize=12)
    ax.set_title('Tweet Volume vs. Effect Size\n(Bubble size = number of influencers)', 
                 fontweight='bold', fontsize=14, pad=20)
    
    # Add labels for key keywords
    for _, row in df_plot.iterrows():
        if row['significant'] and (row['odds_ratio'] > 5 or row['odds_ratio'] < 0.1 or row['total_tweets'] > 20000):
            ax.annotate(row['keyword'], (row['total_tweets'], row['odds_ratio']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
    
    ax.legend(loc='upper left', framealpha=0.9)
    
    # Add annotations
    ax.annotate('Pro-ruling more likely to favor ↓', xy=(0.98, 0.02), xycoords='axes fraction',
                fontsize=10, ha='right', va='bottom', color=COLORS['pro_ruling'], fontweight='bold')
    ax.annotate('Opposition more likely to favor ↑', xy=(0.98, 0.98), xycoords='axes fraction',
                fontsize=10, ha='right', va='top', color=COLORS['opposition'], fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {os.path.basename(output_path)}")


def plot_summary_dashboard(df, output_path):
    """
    Create a summary dashboard with multiple panels.
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Panel 1: Significance count pie chart
    ax1 = fig.add_subplot(gs[0, 0])
    sig_counts = df['significant'].value_counts()
    colors_pie = [COLORS['pro_ruling'], COLORS['not_significant']]
    labels = [f"Significant\n(n={sig_counts.get(True, 0)})", 
              f"Not Significant\n(n={sig_counts.get(False, 0)})"]
    ax1.pie(sig_counts, labels=labels, colors=colors_pie, autopct='%1.1f%%',
            startangle=90, explode=(0.05, 0), textprops={'fontsize': 11})
    ax1.set_title('Significance Distribution', fontweight='bold', fontsize=13)
    
    # Panel 2: Direction of significant effects
    ax2 = fig.add_subplot(gs[0, 1])
    sig_df = df[df['significant']]
    direction_counts = {
        'Pro-ruling favors': (sig_df['beta1'] < 0).sum(),
        'Opposition favors': (sig_df['beta1'] > 0).sum()
    }
    bars = ax2.bar(direction_counts.keys(), direction_counts.values(), 
                   color=[COLORS['pro_ruling'], COLORS['opposition']], edgecolor='white', linewidth=2)
    ax2.set_ylabel('Number of Keywords', fontweight='bold')
    ax2.set_title('Direction of Significant Effects', fontweight='bold', fontsize=13)
    for bar, count in zip(bars, direction_counts.values()):
        ax2.annotate(str(count), xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Panel 3: Top 10 strongest effects (by absolute beta1)
    ax3 = fig.add_subplot(gs[1, 0])
    top_effects = df[df['significant']].copy()
    top_effects['abs_beta'] = top_effects['beta1'].abs()
    top_effects = top_effects.nlargest(10, 'abs_beta').sort_values('beta1', ascending=True)
    
    colors_bar = [COLORS['pro_ruling'] if b < 0 else COLORS['opposition'] for b in top_effects['beta1']]
    ax3.barh(top_effects['keyword'], top_effects['beta1'], color=colors_bar, edgecolor='white')
    ax3.axvline(x=0, color='#6C757D', linestyle='-', linewidth=1)
    ax3.set_xlabel('β₁ Coefficient', fontweight='bold')
    ax3.set_title('Top 10 Strongest Effects', fontweight='bold', fontsize=13)
    
    # Panel 4: Tweet volume by significance
    ax4 = fig.add_subplot(gs[1, 1])
    sig_tweets = df[df['significant']]['total_tweets'].sum()
    nonsig_tweets = df[~df['significant']]['total_tweets'].sum()
    
    bars = ax4.bar(['Significant\nKeywords', 'Non-significant\nKeywords'], 
                   [sig_tweets, nonsig_tweets],
                   color=[COLORS['opposition'], COLORS['not_significant']], edgecolor='white', linewidth=2)
    ax4.set_ylabel('Total Tweets', fontweight='bold')
    ax4.set_title('Tweet Volume Coverage', fontweight='bold', fontsize=13)
    ax4.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    for bar, count in zip(bars, [sig_tweets, nonsig_tweets]):
        ax4.annotate(f'{count:,}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    fig.suptitle('Binomial GLM Significance Testing Summary', fontweight='bold', fontsize=16, y=0.98)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {os.path.basename(output_path)}")


def main():
    """Main execution."""
    print("=" * 60)
    print("VISUALIZING BINOMIAL GLM RESULTS")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    df = load_results()
    
    # Setup style
    setup_plot_style()
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    plot_forest(df, os.path.join(OUTPUT_DIR, "1_forest_plot_odds_ratio.png"))
    plot_coefficient_bars(df, os.path.join(OUTPUT_DIR, "2_coefficient_bar_chart.png"))
    plot_heatmap(df, os.path.join(OUTPUT_DIR, "3_effect_heatmap.png"))
    plot_bubble_chart(df, os.path.join(OUTPUT_DIR, "4_bubble_chart_volume_effect.png"))
    plot_summary_dashboard(df, os.path.join(OUTPUT_DIR, "5_summary_dashboard.png"))
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
