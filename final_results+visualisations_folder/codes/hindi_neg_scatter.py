#!/usr/bin/env python3
"""
Hindi vs English Negativity Scatter Plot

This script creates a scatter plot comparing keyword-level negativity
between Hindi and English tweets for Pro-Ruling and Pro-Opposition parties.

Based on plot_6_scatter_comparison from hindi_english_stance_comparison.py
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
plt.rcParams.update({
    'figure.figsize': (16, 10),
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'axes.titlesize': 14,
    'axes.labelsize': 12
})

# Color schemes
STANCE_COLORS = {
    'favor': '#27ae60',     # Green
    'against': '#c0392b',   # Red  
    'neutral': '#7f8c8d'    # Gray
}

LANGUAGE_COLORS = {
    'hindi': '#8e44ad',     # Purple
    'english': '#2980b9'    # Blue
}


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
    df['language'] = df['language'].str.lower().str.strip()  # Standardize language column
    
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


def plot_6_scatter_comparison(df, output_dir):
    """
    Scatter plot: Each point is a keyword. X-axis = English against %,
    Y-axis = Hindi against %. Points above the diagonal = more negative in Hindi.
    """
    print("\nGenerating Plot 6: Hindi vs English Negativity Scatter...")
    
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
                hi_against = (hi_subset['stance'] == 'against').sum() / len(hi_subset) * 100
                en_against = (en_subset['stance'] == 'against').sum() / len(en_subset) * 100
                points.append({'Keyword': kw, 'Hindi': hi_against, 'English': en_against})
        
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
        ax.plot([0, 80], [0, 80], 'k--', alpha=0.5, linewidth=2, label='Equal negativity')
        
        # Scatter plot
        ax.scatter(scatter_df['English'], scatter_df['Hindi'], 
                  s=120, c='#e74c3c', alpha=0.7, edgecolors='white', linewidth=1)
        
        # Label points
        for _, row in scatter_df.iterrows():
            ax.annotate(row['Keyword'], (row['English'], row['Hindi']),
                       xytext=(5, 5), textcoords='offset points', fontsize=11, alpha=0.8)
        
        ax.set_xlabel('English "Against" %', fontweight='bold', fontsize=12)
        ax.set_ylabel('Hindi "Against" %', fontweight='bold', fontsize=12)
        # Subplot title - just show party label (no description, LaTeX will caption)
        ax.set_title(f'{party_label}', fontweight='bold', fontsize=14)
        ax.set_xlim(0, 75)
        ax.set_ylim(0, 75)
        ax.legend(loc='upper left')
        
        # Add annotation for interpretation
        ax.fill_between([0, 80], [0, 80], [80, 80], alpha=0.1, color='red')
        ax.fill_between([0, 80], [0, 0], [0, 80], alpha=0.1, color='green')
        ax.text(10, 65, 'More negative\nin Hindi', fontsize=10, color='darkred', alpha=0.7)
        ax.text(55, 10, 'More negative\nin English', fontsize=10, color='darkgreen', alpha=0.7)
    
    # Title/suptitle removed for LaTeX (caption will be in LaTeX document)
    plt.tight_layout()
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as PDF for LaTeX compatibility (as per figure guidelines)
    output_path = output_dir / '6_negativity_scatter.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"  Saved: {output_path}")
    
    metrics_path = output_dir / '6_negativity_scatter_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write("Distances are perpendicular to the diagonal y=x, in percentage points.\n")
        f.write("Metrics: RMSE, MAE, MedianAbs, MeanSigned.\n\n")
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


def main():
    """Main function to generate the visualization."""
    print("=" * 60)
    print("HINDI vs ENGLISH NEGATIVITY SCATTER PLOT")
    print("=" * 60)
    
    # Set output directory
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput: {output_dir}\n")
    
    # Load and preprocess
    df = load_and_preprocess()
    
    # Generate visualization
    plot_6_scatter_comparison(df, output_dir)
    
    print("\n" + "=" * 60)
    print("COMPLETE! Visualization saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()
