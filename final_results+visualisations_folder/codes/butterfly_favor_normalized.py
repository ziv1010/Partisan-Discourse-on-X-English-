#!/usr/bin/env python3
"""
Butterfly Favor Normalized Visualization

This script creates a butterfly chart comparing normalized favor percentages
between Pro-Ruling and Pro-Opposition parties for each keyword.

Based on plot_butterfly_polarization from test.ipynb
(normalized_analysis_v3_compact)
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

CSV_PATH = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/final_results+visualisations_folder/combined_stance_results.csv"

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')

# Colors
PARTY_COLORS = {
    'pro ruling': '#FF6B35',
    'pro opposition': '#004E89'
}

# Font sizes
FS_TITLE = 16
FS_LABEL = 12
FS_TICK = 10

# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

def load_data():
    """Load and preprocess the stance data."""
    if not Path(CSV_PATH).exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")
    
    print(f"Loading data from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    print(f"✓ Loaded {len(df):,} rows")
    
    # Filter and Normalize stance labels
    df = df[df['fewshot_label'].isin(['favor', 'against', 'neutral'])].copy()
    df['stance'] = df['fewshot_label']
    df['party'] = df['_label_norm'].str.lower().str.strip()
    df['keyword'] = df['keyword'].str.lower().str.strip()
    
    df = df[df['party'].isin(['pro ruling', 'pro opposition'])]
    
    print(f"✓ Filtered to {len(df):,} valid tweets")
    return df


def plot_butterfly_polarization(df, output_dir):
    """
    Create a butterfly chart showing normalized favor comparison 
    between Ruling and Opposition parties.
    """
    print("\nGenerating Butterfly Favor Normalized chart...")
    
    # Calculate normalized stance percentages
    stats = df.groupby(['keyword', 'party'])['stance'].value_counts(normalize=True).unstack(fill_value=0) * 100
    counts = df.groupby(['keyword', 'party']).size().unstack(fill_value=0)
    
    # Filter to keywords with sufficient data in both parties
    valid_kws = counts[(counts['pro ruling'] > 15) & (counts['pro opposition'] > 15)].index
    
    if valid_kws.empty:
        print("  No valid keywords found!")
        return
    
    print(f"  Found {len(valid_kws)} keywords with sufficient data")
    
    plot_df = stats.loc[valid_kws].reset_index()
    ruling = plot_df[plot_df['party'] == 'pro ruling'].set_index('keyword')
    opp = plot_df[plot_df['party'] == 'pro opposition'].set_index('keyword')
    
    # Calculate difference and sort
    diff = (ruling['favor'] - opp['favor']).sort_values()
    
    # Create figure with dynamic height based on number of keywords
    fig_height = len(diff) * 0.4 + 2
    fig, ax = plt.subplots(figsize=(12, fig_height))
    
    y_pos = np.arange(len(diff))
    bar_height = 0.7
    
    # Plot bars for both parties (opposition on left, ruling on right)
    ax.barh(y_pos, -opp.loc[diff.index, 'favor'], 
            color=PARTY_COLORS['pro opposition'], 
            label='Opposition Favor %', 
            height=bar_height)
    ax.barh(y_pos, ruling.loc[diff.index, 'favor'], 
            color=PARTY_COLORS['pro ruling'], 
            label='Ruling Favor %', 
            height=bar_height)
    
    # Configure axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(diff.index, fontsize=FS_TICK)
    ax.axvline(0, color='black', lw=1)
    
    # Labels and title
    plt.title("Normalized Favor Comparison: Ruling vs Opposition", fontsize=FS_TITLE, fontweight='bold')
    plt.xlabel("← Opposition % Favor | Ruling % Favor →", fontsize=FS_LABEL, fontweight='bold')
    plt.xticks(fontsize=FS_TICK)
    plt.legend(fontsize=FS_TICK, loc='lower right')
    
    plt.tight_layout()
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / '2_butterfly_favor_normalized.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def main():
    """Main function to generate the visualization."""
    print("=" * 60)
    print("BUTTERFLY FAVOR NORMALIZED VISUALIZATION")
    print("=" * 60)
    
    # Set output directory
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput: {output_dir}\n")
    
    # Load and preprocess
    df = load_data()
    
    # Generate visualization
    plot_butterfly_polarization(df, output_dir)
    
    print("\n" + "=" * 60)
    print("COMPLETE! Visualization saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()
