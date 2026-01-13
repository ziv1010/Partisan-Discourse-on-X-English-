#!/usr/bin/env python3
"""
Hindi vs English Positivity and Neutral Scatter Plots

This script creates scatter plots comparing keyword-level positivity (favor)
and neutral stances between Hindi and English tweets for Pro-Ruling and Pro-Opposition parties.

Based on hindi_neg_scatter.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

HINDI_CSV = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/hindi_codes/hindi_stance_results.csv"
ENGLISH_CSV = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/6_stance/results/final_en_results/stance_results_37keywords.csv"
COMBINED_CSV = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/final_results+visualisations_folder/combined_stance_results.csv"

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


def load_and_preprocess():
    """Load and preprocess both datasets."""
    print("Loading data...")
    
    # Try to use combined CSV first
    if Path(COMBINED_CSV).exists():
        print(f"Loading from combined CSV: {COMBINED_CSV}")
        df = pd.read_csv(COMBINED_CSV, low_memory=False)
    else:
        # Load Hindi
        hindi_df = pd.read_csv(HINDI_CSV, low_memory=False)
        hindi_df['language'] = 'hindi'
        
        # Load English
        english_df = pd.read_csv(ENGLISH_CSV, low_memory=False)
        english_df['language'] = 'english'
        
        # Combine
        df = pd.concat([hindi_df, english_df], ignore_index=True)
    
    # Filter to valid stances
    valid_stances = ['favor', 'against', 'neutral']
    df = df[df['fewshot_label'].isin(valid_stances)].copy()
    
    # Standardize columns
    df['stance'] = df['fewshot_label']
    df['party'] = df['_label_norm'].str.lower().str.strip()
    df['keyword'] = df['keyword'].str.lower().str.strip()
    df['language'] = df['language'].str.lower().str.strip()
    
    # Filter to only pro ruling and pro opposition
    df = df[df['party'].isin(['pro ruling', 'pro opposition'])]
    
    print(f"Loaded {len(df):,} total tweets")
    print(f"  Hindi: {len(df[df['language']=='hindi']):,}")
    print(f"  English: {len(df[df['language']=='english']):,}")
    
    return df


def get_common_keywords(df, min_samples=100):
    """Get keywords present in both languages with sufficient samples."""
    keywords = []
    for kw in df['keyword'].unique():
        hi_count = len(df[(df['keyword']==kw) & (df['language']=='hindi')])
        en_count = len(df[(df['keyword']==kw) & (df['language']=='english')])
        if hi_count >= min_samples and en_count >= min_samples:
            keywords.append(kw)
    return sorted(keywords)


def plot_stance_scatter(df, output_dir, stance_type, color, label_above, label_below):
    """
    Generic scatter plot for a given stance type.
    X-axis = English stance %, Y-axis = Hindi stance %.
    """
    stance_title = stance_type.capitalize()
    print(f"\nGenerating {stance_title} Scatter Plot...")
    
    common_keywords = get_common_keywords(df)
    print(f"  Found {len(common_keywords)} common keywords")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    metrics_rows = []
    
    for idx, party in enumerate(['pro ruling', 'pro opposition']):
        ax = axes[idx]
        party_label = 'Pro-Ruling' if party == 'pro ruling' else 'Pro-Opposition'
        
        points = []
        for kw in common_keywords:
            hi_subset = df[(df['keyword']==kw) & (df['party']==party) & (df['language']=='hindi')]
            en_subset = df[(df['keyword']==kw) & (df['party']==party) & (df['language']=='english')]
            
            if len(hi_subset) > 20 and len(en_subset) > 20:
                hi_pct = (hi_subset['stance'] == stance_type).sum() / len(hi_subset) * 100
                en_pct = (en_subset['stance'] == stance_type).sum() / len(en_subset) * 100
                points.append({'Keyword': kw, 'Hindi': hi_pct, 'English': en_pct})
        
        scatter_df = pd.DataFrame(points)
        
        if len(scatter_df) == 0:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            metrics_rows.append({'party': party_label, 'n_points': 0})
            continue
        
        diff = scatter_df['Hindi'] - scatter_df['English']
        perp = diff / np.sqrt(2)
        metrics_rows.append({
            'party': party_label,
            'n_points': len(scatter_df),
            'rmse': np.sqrt(np.mean(perp ** 2)),
            'mae': np.mean(np.abs(perp)),
            'median_abs': np.median(np.abs(perp)),
            'mean_signed': np.mean(perp)
        })
        
        # Plot diagonal line
        ax.plot([0, 100], [0, 100], 'k--', alpha=0.5, linewidth=2, label='Equal stance')
        
        # Scatter plot
        ax.scatter(scatter_df['English'], scatter_df['Hindi'], 
                  s=120, c=color, alpha=0.7, edgecolors='white', linewidth=1)
        
        # Label points
        for _, row in scatter_df.iterrows():
            ax.annotate(row['Keyword'], (row['English'], row['Hindi']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
        
        ax.set_xlabel(f'English "{stance_title}" %', fontweight='bold', fontsize=12)
        ax.set_ylabel(f'Hindi "{stance_title}" %', fontweight='bold', fontsize=12)
        ax.set_title(f'{party_label}:\nKeyword {stance_title} Comparison', fontweight='bold', fontsize=14)
        
        # Set limits based on data
        max_val = max(scatter_df['Hindi'].max(), scatter_df['English'].max()) + 10
        max_val = min(max_val, 100)
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.legend(loc='upper left')
        
        # Add annotation for interpretation
        ax.fill_between([0, max_val], [0, max_val], [max_val, max_val], alpha=0.1, color='purple')
        ax.fill_between([0, max_val], [0, 0], [0, max_val], alpha=0.1, color='blue')
        ax.text(5, max_val-10, label_above, fontsize=10, color='purple', alpha=0.7)
        ax.text(max_val-30, 5, label_below, fontsize=10, color='blue', alpha=0.7)
    
    plt.suptitle(f'Keyword-Level {stance_title}: Hindi vs English\n(Each point = one keyword)', 
                 fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f'{stance_type}_scatter.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    
    metrics_path = output_dir / f'{stance_type}_scatter_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write(f"Distances are perpendicular to the diagonal y=x, in percentage points.\n")
        f.write(f"Metrics: RMSE, MAE, MedianAbs, MeanSigned.\n\n")
        for row in metrics_rows:
            f.write(f"{row['party']} (n={row['n_points']} keywords)\n")
            if row['n_points'] == 0:
                f.write("  No data available\n\n")
                continue
            f.write(f"  RMSE: {row['rmse']:.3f}\n")
            f.write(f"  MAE: {row['mae']:.3f}\n")
            f.write(f"  MedianAbs: {row['median_abs']:.3f}\n")
            f.write(f"  MeanSigned: {row['mean_signed']:.3f}\n\n")
    print(f"  Metrics saved: {metrics_path}")
    
    return metrics_rows


def main():
    """Main function to generate the visualizations."""
    print("=" * 60)
    print("HINDI vs ENGLISH: POSITIVITY & NEUTRAL SCATTER PLOTS")
    print("=" * 60)
    
    # Set output directory
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput: {output_dir}\n")
    
    # Load and preprocess
    df = load_and_preprocess()
    
    # Generate Positivity (Favor) scatter plot
    favor_metrics = plot_stance_scatter(
        df, output_dir, 
        stance_type='favor',
        color='#27ae60',  # Green
        label_above='More positive\nin Hindi',
        label_below='More positive\nin English'
    )
    
    # Generate Neutral scatter plot
    neutral_metrics = plot_stance_scatter(
        df, output_dir,
        stance_type='neutral',
        color='#7f8c8d',  # Gray
        label_above='More neutral\nin Hindi',
        label_below='More neutral\nin English'
    )
    
    # Print summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY: COMPARING HINDI VS ENGLISH CONSISTENCY")
    print("=" * 60)
    print("\nLower RMSE/MAE values = more consistent across languages\n")
    
    print("FAVOR (Positivity):")
    for row in favor_metrics:
        if row['n_points'] > 0:
            print(f"  {row['party']}: RMSE={row['rmse']:.2f}, MAE={row['mae']:.2f}")
    
    print("\nNEUTRAL:")
    for row in neutral_metrics:
        if row['n_points'] > 0:
            print(f"  {row['party']}: RMSE={row['rmse']:.2f}, MAE={row['mae']:.2f}")
    
    print("\n(Compare with negativity RMSE/MAE from hindi_neg_scatter.py)")
    print("\n" + "=" * 60)
    print("COMPLETE! Visualizations saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()
