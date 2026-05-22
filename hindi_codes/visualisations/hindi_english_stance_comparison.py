#!/usr/bin/env python3
"""
Hindi vs English Stance Comparison: Keyword-Based Analysis

Research Question: Do parties tend to use Hindi more for negative tweets 
compared to using English?

Focus: Keyword-based comparisons between Pro-Ruling and Pro-Opposition 
across Hindi and English languages.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

HINDI_CSV = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/hindi_codes/hindi_stance_results.csv"
ENGLISH_CSV = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/6_stance/results/final_en_results/stance_results_37keywords.csv"
OUTPUT_DIR = Path("/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/hindi_codes/visualisations/output")

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Color schemes
STANCE_COLORS = {
    'favor': '#27ae60',     # Green
    'against': '#c0392b',   # Red  
    'neutral': '#7f8c8d'    # Gray
}

LANGUAGE_COLORS = {
    'Hindi': '#8e44ad',     # Purple
    'English': '#2980b9'    # Blue
}


def load_and_preprocess():
    """Load and preprocess both datasets."""
    print("Loading data...")
    
    # Load Hindi
    hindi_df = pd.read_csv(HINDI_CSV, low_memory=False)
    hindi_df['language'] = 'Hindi'
    
    # Load English
    english_df = pd.read_csv(ENGLISH_CSV, low_memory=False)
    english_df['language'] = 'English'
    
    # Combine
    df = pd.concat([hindi_df, english_df], ignore_index=True)
    
    # Filter to valid stances
    valid_stances = ['favor', 'against', 'neutral']
    df = df[df['fewshot_label'].isin(valid_stances)].copy()
    
    # Standardize columns
    df['stance'] = df['fewshot_label']
    df['party'] = df['_label_norm'].str.lower().str.strip()
    df['keyword'] = df['keyword'].str.lower().str.strip()
    
    # Filter to only pro ruling and pro opposition
    df = df[df['party'].isin(['pro ruling', 'pro opposition'])]
    
    print(f"Loaded {len(df):,} total tweets")
    print(f"  Hindi: {len(df[df['language']=='Hindi']):,}")
    print(f"  English: {len(df[df['language']=='English']):,}")
    
    return df


def get_common_keywords(df, min_samples=100):
    """Get keywords present in both languages with sufficient samples."""
    keywords = []
    for kw in df['keyword'].unique():
        hi_count = len(df[(df['keyword']==kw) & (df['language']=='Hindi')])
        en_count = len(df[(df['keyword']==kw) & (df['language']=='English')])
        if hi_count >= min_samples and en_count >= min_samples:
            keywords.append(kw)
    return sorted(keywords)


def plot_1_negativity_by_keyword(df, output_dir):
    """
    For each keyword: Compare % of "against" tweets in Hindi vs English,
    split by Pro-Ruling and Pro-Opposition.
    
    This directly answers: Do parties use Hindi more for negative tweets?
    """
    print("\nGenerating Plot 1: Negativity by Keyword...")
    
    common_keywords = get_common_keywords(df)
    
    # Calculate against % for each combination
    data = []
    for kw in common_keywords:
        for party in ['pro ruling', 'pro opposition']:
            for lang in ['Hindi', 'English']:
                subset = df[(df['keyword']==kw) & (df['party']==party) & (df['language']==lang)]
                if len(subset) > 20:
                    against_pct = (subset['stance'] == 'against').sum() / len(subset) * 100
                    data.append({
                        'Keyword': kw,
                        'Party': 'Pro-Ruling' if party == 'pro ruling' else 'Pro-Opposition',
                        'Language': lang,
                        'Against %': against_pct,
                        'Count': len(subset)
                    })
    
    plot_df = pd.DataFrame(data)
    
    # Create side-by-side bar charts
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    for idx, party in enumerate(['Pro-Ruling', 'Pro-Opposition']):
        ax = axes[idx]
        party_data = plot_df[plot_df['Party'] == party]
        
        # Pivot for plotting
        pivot = party_data.pivot(index='Keyword', columns='Language', values='Against %')
        pivot = pivot.dropna()
        pivot = pivot.sort_values('Hindi', ascending=True)
        
        x = np.arange(len(pivot))
        width = 0.35
        
        bars1 = ax.barh(x - width/2, pivot['Hindi'], width, 
                       label='Hindi', color=LANGUAGE_COLORS['Hindi'], alpha=0.9)
        bars2 = ax.barh(x + width/2, pivot['English'], width, 
                       label='English', color=LANGUAGE_COLORS['English'], alpha=0.9)
        
        ax.set_yticks(x)
        ax.set_yticklabels(pivot.index, fontsize=10)
        ax.set_xlabel('Negative Tweets (Against %)', fontweight='bold', fontsize=12)
        ax.set_title(f'{party}:\n"Against" Stance by Keyword & Language', 
                    fontweight='bold', fontsize=14)
        ax.legend(loc='lower right', fontsize=11)
        ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
        
        # Add value labels
        for bar in bars1:
            width_val = bar.get_width()
            ax.annotate(f'{width_val:.0f}%',
                       xy=(width_val, bar.get_y() + bar.get_height()/2),
                       xytext=(3, 0), textcoords="offset points",
                       ha='left', va='center', fontsize=8, color=LANGUAGE_COLORS['Hindi'])
        for bar in bars2:
            width_val = bar.get_width()
            ax.annotate(f'{width_val:.0f}%',
                       xy=(width_val, bar.get_y() + bar.get_height()/2),
                       xytext=(3, 0), textcoords="offset points",
                       ha='left', va='center', fontsize=8, color=LANGUAGE_COLORS['English'])
    
    plt.suptitle('Do Parties Use Hindi More for Negative Tweets?\nComparing "Against" Stance by Keyword', 
                 fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / '1_negativity_by_keyword.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 1_negativity_by_keyword.png")


def plot_2_hindi_vs_english_difference(df, output_dir):
    """
    Butterfly chart: For each keyword, show the DIFFERENCE (Hindi - English)
    in "against" percentage. Positive = more negative in Hindi.
    """
    print("\nGenerating Plot 2: Hindi-English Negativity Difference...")
    
    common_keywords = get_common_keywords(df)
    
    data = []
    for kw in common_keywords:
        for party in ['pro ruling', 'pro opposition']:
            hi_subset = df[(df['keyword']==kw) & (df['party']==party) & (df['language']=='Hindi')]
            en_subset = df[(df['keyword']==kw) & (df['party']==party) & (df['language']=='English')]
            
            if len(hi_subset) > 20 and len(en_subset) > 20:
                hi_against = (hi_subset['stance'] == 'against').sum() / len(hi_subset) * 100
                en_against = (en_subset['stance'] == 'against').sum() / len(en_subset) * 100
                diff = hi_against - en_against
                
                data.append({
                    'Keyword': kw,
                    'Party': 'Pro-Ruling' if party == 'pro ruling' else 'Pro-Opposition',
                    'Hindi Against %': hi_against,
                    'English Against %': en_against,
                    'Difference': diff
                })
    
    diff_df = pd.DataFrame(data)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    
    for idx, party in enumerate(['Pro-Ruling', 'Pro-Opposition']):
        ax = axes[idx]
        party_data = diff_df[diff_df['Party'] == party].sort_values('Difference')
        
        colors = ['#27ae60' if d < 0 else '#c0392b' for d in party_data['Difference']]
        
        bars = ax.barh(range(len(party_data)), party_data['Difference'], color=colors, 
                      edgecolor='white', linewidth=0.5)
        
        ax.set_yticks(range(len(party_data)))
        ax.set_yticklabels(party_data['Keyword'], fontsize=10)
        ax.axvline(x=0, color='black', linewidth=1.5)
        ax.set_xlabel('Difference in "Against" % (Hindi − English)', fontweight='bold', fontsize=12)
        ax.set_title(f'{party}:\nLanguage Negativity Difference by Keyword', fontweight='bold', fontsize=14)
        
        # Add annotations
        for i, (bar, row) in enumerate(zip(bars, party_data.itertuples())):
            width = bar.get_width()
            label = f'{width:+.1f}%'
            ax.annotate(label,
                       xy=(width, bar.get_y() + bar.get_height() / 2),
                       xytext=(5 if width >= 0 else -5, 0),
                       textcoords="offset points",
                       ha='left' if width >= 0 else 'right',
                       va='center', fontsize=9, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#c0392b', label='More negative in Hindi'),
        Patch(facecolor='#27ae60', label='More negative in English')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=12, 
              bbox_to_anchor=(0.5, 0.02))
    
    plt.suptitle('Language Strategy Difference:\nWhere Do Parties Use More Negativity?', 
                 fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(output_dir / '2_negativity_difference_butterfly.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 2_negativity_difference_butterfly.png")


def plot_3_stance_heatmap_comparison(df, output_dir):
    """
    Side-by-side heatmaps: Full stance distribution (favor/against/neutral)
    for top keywords, comparing Hindi vs English by party.
    """
    print("\nGenerating Plot 3: Stance Heatmap Comparison...")
    
    top_keywords = df['keyword'].value_counts().head(15).index.tolist()
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    
    for row_idx, party in enumerate(['pro ruling', 'pro opposition']):
        for col_idx, lang in enumerate(['Hindi', 'English']):
            ax = axes[row_idx, col_idx]
            
            # Calculate stance % for each keyword
            heatmap_data = []
            for kw in top_keywords:
                subset = df[(df['keyword']==kw) & (df['party']==party) & (df['language']==lang)]
                if len(subset) > 10:
                    row = {'Keyword': f'{kw} (n={len(subset):,})'}
                    for stance in ['favor', 'neutral', 'against']:
                        row[stance.capitalize()] = (subset['stance'] == stance).sum() / len(subset) * 100
                    heatmap_data.append(row)
            
            heat_df = pd.DataFrame(heatmap_data).set_index('Keyword')
            
            if len(heat_df) > 0:
                # Get total count for this party/language combo
                total_n = len(df[(df['party']==party) & (df['language']==lang)])
                
                sns.heatmap(heat_df, annot=True, fmt='.0f', cmap='RdYlGn_r', 
                           center=33.3, vmin=0, vmax=70, linewidths=0.5,
                           cbar_kws={'label': '%'}, ax=ax)
                
                party_label = 'Pro-Ruling' if party == 'pro ruling' else 'Pro-Opposition'
                ax.set_title(f'{party_label} - {lang}\n(Total: {total_n:,} tweets)', 
                            fontweight='bold', fontsize=12)
                ax.set_xlabel('Stance', fontweight='bold')
                ax.set_ylabel('Keyword (with tweet count)', fontweight='bold')
    
    plt.suptitle('Stance Distribution by Keyword:\nComparing Languages & Party Affiliations', 
                 fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / '3_stance_heatmap_4way.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 3_stance_heatmap_4way.png")


def plot_4_top_keywords_comparison(df, output_dir):
    """
    For top 10 keywords: Grouped bar chart comparing stance distribution
    between Hindi and English for each party.
    """
    print("\nGenerating Plot 4: Top Keywords Detailed Comparison...")
    
    top_kws = ['modi', 'congress', 'hindu', 'rahulgandhi', 'muslim', 
               'kashmir', 'china', 'ucc', 'democracy', 'ayodhya']
    
    fig, axes = plt.subplots(5, 2, figsize=(16, 20))
    axes = axes.flatten()
    
    for i, kw in enumerate(top_kws):
        ax = axes[i]
        
        # Prepare data
        plot_data = []
        for party in ['pro ruling', 'pro opposition']:
            for lang in ['Hindi', 'English']:
                subset = df[(df['keyword']==kw) & (df['party']==party) & (df['language']==lang)]
                if len(subset) > 0:
                    for stance in ['favor', 'against', 'neutral']:
                        pct = (subset['stance'] == stance).sum() / len(subset) * 100
                        party_label = 'Pro-Ruling' if party == 'pro ruling' else 'Pro-Opp'
                        plot_data.append({
                            'Group': f'{party_label}\n{lang}',
                            'Stance': stance.capitalize(),
                            'Percentage': pct
                        })
        
        if plot_data:
            pdata = pd.DataFrame(plot_data)
            pivot = pdata.pivot(index='Group', columns='Stance', values='Percentage')
            pivot = pivot[['Favor', 'Neutral', 'Against']]
            
            pivot.plot(kind='bar', stacked=True, ax=ax, 
                      color=[STANCE_COLORS['favor'], STANCE_COLORS['neutral'], STANCE_COLORS['against']],
                      edgecolor='white', linewidth=0.5, legend=(i==0))
            
            ax.set_title(f'Keyword: {kw.upper()}', fontweight='bold', fontsize=11)
            ax.set_xlabel('')
            ax.set_ylabel('Percentage (%)')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
            ax.set_ylim(0, 100)
            
            if i == 0:
                ax.legend(title='Stance', bbox_to_anchor=(1.0, 1.0), fontsize=9)
    
    plt.suptitle('Stance Breakdown by Keyword:\nPro-Ruling vs Pro-Opposition × Hindi vs English', 
                 fontweight='bold', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / '4_top_keywords_stacked.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 4_top_keywords_stacked.png")


def plot_5_party_language_strategy(df, output_dir):
    """
    Direct comparison: For each party, how does their stance differ between
    Hindi and English across all keywords?
    """
    print("\nGenerating Plot 5: Party Language Strategy Overview...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    for idx, party in enumerate(['pro ruling', 'pro opposition']):
        ax = axes[idx]
        party_label = 'Pro-Ruling' if party == 'pro ruling' else 'Pro-Opposition'
        
        # Calculate overall stance by language
        plot_data = []
        lang_counts = {}
        for lang in ['Hindi', 'English']:
            subset = df[(df['party']==party) & (df['language']==lang)]
            n = len(subset)
            lang_counts[lang] = n
            for stance in ['favor', 'against', 'neutral']:
                pct = (subset['stance'] == stance).sum() / n * 100
                plot_data.append({
                    'Language': lang,
                    'Stance': stance.capitalize(),
                    'Percentage': pct
                })
        
        pdata = pd.DataFrame(plot_data)
        pivot = pdata.pivot(index='Language', columns='Stance', values='Percentage')
        pivot = pivot[['Favor', 'Neutral', 'Against']]
        
        x = np.arange(len(pivot.columns))
        width = 0.35
        
        hindi_vals = pivot.loc['Hindi'].values
        english_vals = pivot.loc['English'].values
        
        bars1 = ax.bar(x - width/2, hindi_vals, width, 
                      label=f'Hindi (n={lang_counts["Hindi"]:,})', 
                      color=LANGUAGE_COLORS['Hindi'], edgecolor='white')
        bars2 = ax.bar(x + width/2, english_vals, width, 
                      label=f'English (n={lang_counts["English"]:,})', 
                      color=LANGUAGE_COLORS['English'], edgecolor='white')
        
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.columns, fontsize=12)
        ax.set_ylabel('Percentage (%)', fontweight='bold')
        ax.set_title(f'{party_label}:\nStance Distribution by Language', fontweight='bold', fontsize=14)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 55)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle('Language Strategy Summary:\nDo Parties Tweet More Negatively in Hindi vs English?', 
                 fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / '5_party_language_strategy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 5_party_language_strategy.png")


def plot_6_scatter_comparison(df, output_dir):
    """
    Scatter plot: Each point is a keyword. X-axis = English against %,
    Y-axis = Hindi against %. Points above the diagonal = more negative in Hindi.
    """
    print("\nGenerating Plot 6: Hindi vs English Negativity Scatter...")
    
    common_keywords = get_common_keywords(df)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    for idx, party in enumerate(['pro ruling', 'pro opposition']):
        ax = axes[idx]
        party_label = 'Pro-Ruling' if party == 'pro ruling' else 'Pro-Opposition'
        
        points = []
        for kw in common_keywords:
            hi_subset = df[(df['keyword']==kw) & (df['party']==party) & (df['language']=='Hindi')]
            en_subset = df[(df['keyword']==kw) & (df['party']==party) & (df['language']=='English')]
            
            if len(hi_subset) > 20 and len(en_subset) > 20:
                hi_against = (hi_subset['stance'] == 'against').sum() / len(hi_subset) * 100
                en_against = (en_subset['stance'] == 'against').sum() / len(en_subset) * 100
                points.append({'Keyword': kw, 'Hindi': hi_against, 'English': en_against})
        
        scatter_df = pd.DataFrame(points)
        
        # Plot diagonal line
        ax.plot([0, 80], [0, 80], 'k--', alpha=0.5, linewidth=2, label='Equal negativity')
        
        # Scatter plot
        ax.scatter(scatter_df['English'], scatter_df['Hindi'], 
                  s=120, c='#e74c3c', alpha=0.7, edgecolors='white', linewidth=1)
        
        # Label points
        for _, row in scatter_df.iterrows():
            ax.annotate(row['Keyword'], (row['English'], row['Hindi']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
        
        ax.set_xlabel('English "Against" %', fontweight='bold', fontsize=12)
        ax.set_ylabel('Hindi "Against" %', fontweight='bold', fontsize=12)
        ax.set_title(f'{party_label}:\nKeyword Negativity Comparison', fontweight='bold', fontsize=14)
        ax.set_xlim(0, 75)
        ax.set_ylim(0, 75)
        ax.legend(loc='upper left')
        
        # Add annotation for interpretation
        ax.fill_between([0, 80], [0, 80], [80, 80], alpha=0.1, color='red')
        ax.fill_between([0, 80], [0, 0], [0, 80], alpha=0.1, color='green')
        ax.text(10, 65, 'More negative\nin Hindi', fontsize=10, color='darkred', alpha=0.7)
        ax.text(55, 10, 'More negative\nin English', fontsize=10, color='darkgreen', alpha=0.7)
    
    plt.suptitle('Keyword-Level Negativity: Hindi vs English\n(Each point = one keyword)', 
                 fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / '6_negativity_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 6_negativity_scatter.png")


def plot_7_all_keywords_comparison(df, output_dir):
    """
    For ALL keywords: Grouped bar chart comparing stance distribution
    between Hindi and English for each party. Similar to plot 4 but for everything.
    Dynamic height adjustment based on number of keywords.
    """
    print("\nGenerating Plot 7: All Keywords Detailed Comparison...")
    
    common_keywords = get_common_keywords(df, min_samples=50) # Lower threshold to include more
    # If too many, maybe take top 50? No, user asked for "all"
    
    if not common_keywords:
        print("  No keywords found for Plot 7.")
        return

    n_keywords = len(common_keywords)
    print(f"  Found {n_keywords} common keywords to plot.")
    
    # Calculate grid size (2 columns)
    cols = 2
    rows = int(np.ceil(n_keywords / cols))
    
    # Dynamic figure height: approx 4 inches per row
    fig_height = max(10, rows * 4)
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, fig_height))
    axes = axes.flatten()
    
    for i, kw in enumerate(common_keywords):
        ax = axes[i]
        
        # Prepare data
        plot_data = []
        has_data = False
        
        for party in ['pro ruling', 'pro opposition']:
            for lang in ['Hindi', 'English']:
                subset = df[(df['keyword']==kw) & (df['party']==party) & (df['language']==lang)]
                if len(subset) > 0:
                    has_data = True
                    for stance in ['favor', 'against', 'neutral']:
                        pct = (subset['stance'] == stance).sum() / len(subset) * 100
                        party_label = 'Pro-Ruling' if party == 'pro ruling' else 'Pro-Opp'
                        plot_data.append({
                            'Group': f'{party_label}\n{lang}',
                            'Stance': stance.capitalize(),
                            'Percentage': pct
                        })
        
        if has_data:
            pdata = pd.DataFrame(plot_data)
            pivot = pdata.pivot(index='Group', columns='Stance', values='Percentage')
            
            # Ensure all columns exist
            for col in ['Favor', 'Neutral', 'Against']:
                if col not in pivot.columns:
                    pivot[col] = 0
            
            pivot = pivot[['Favor', 'Neutral', 'Against']]
            
            pivot.plot(kind='bar', stacked=True, ax=ax, 
                      color=[STANCE_COLORS['favor'], STANCE_COLORS['neutral'], STANCE_COLORS['against']],
                      edgecolor='white', linewidth=0.5, legend=(i==0))
            
            ax.set_title(f'Keyword: {kw.upper()}', fontweight='bold', fontsize=11)
            ax.set_xlabel('')
            ax.set_ylabel('Percentage (%)')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
            ax.set_ylim(0, 100)
            
            if i == 0:
                ax.legend(title='Stance', bbox_to_anchor=(1.0, 1.0), fontsize=9)
        else:
             ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center')
             ax.axis('off')

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle('Stance Breakdown by Keyword (ALL): \nPro-Ruling vs Pro-Opposition × Hindi vs English', 
                 fontweight='bold', fontsize=16, y=1.00 - (0.5/fig_height)) # Adjust title position based on height
    plt.tight_layout()
    # Adjust top margin to make room for title, depending on how tall the figure is
    plt.subplots_adjust(top=1.0 - (1.0 / fig_height)) 
    
    plt.savefig(output_dir / '7_all_keywords_stacked.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 7_all_keywords_stacked.png")


def main():
    """Main function to generate all visualizations."""
    print("=" * 60)
    print("HINDI vs ENGLISH STANCE COMPARISON")
    print("Research Q: Do parties use Hindi more for negative tweets?")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput: {OUTPUT_DIR}\n")
    
    # Load and preprocess
    df = load_and_preprocess()
    
    # Generate visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    plot_1_negativity_by_keyword(df, OUTPUT_DIR)
    plot_2_hindi_vs_english_difference(df, OUTPUT_DIR)
    plot_3_stance_heatmap_comparison(df, OUTPUT_DIR)
    plot_4_top_keywords_comparison(df, OUTPUT_DIR)
    plot_5_party_language_strategy(df, OUTPUT_DIR)
    plot_6_scatter_comparison(df, OUTPUT_DIR)
    plot_7_all_keywords_comparison(df, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("\n" + "=" * 60)
    print("COMPLETE! 7 visualizations saved.")
    print("=" * 60)
    print("=" * 60)


if __name__ == "__main__":
    main()
