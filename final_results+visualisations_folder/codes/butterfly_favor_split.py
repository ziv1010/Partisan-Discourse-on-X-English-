#!/usr/bin/env python3
"""
Split Butterfly Favor Normalized Visualizations

Creates two separate butterfly charts:
1. Aspects where Pro-Ruling has higher favor % (pro-ruling favored topics)
2. Aspects where Pro-Opposition has higher favor % (pro-opposition favored topics)

Neutral/ambiguous aspects (small favor difference) are excluded.

Normalization: For each keyword and party, the favor % is calculated as:
    (number of "favor" tweets) / (total tweets for that keyword-party combo) * 100

This allows fair comparison between parties with different tweet volumes per topic.
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

# Minimum percentage point difference to be considered non-neutral
# Aspects with |difference| < threshold are considered "neutral" and excluded
NEUTRAL_THRESHOLD = 5.0  # percentage points

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')

# Colors
PARTY_COLORS = {
    'pro ruling': '#FF6B35',
    'pro opposition': '#004E89'
}

# Font sizes
FS_TITLE = 14
FS_SUBTITLE = 11
FS_LABEL = 11
FS_TICK = 9

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


def calculate_favor_stats(df):
    """
    Calculate normalized favor percentages for each keyword and party.
    
    Returns DataFrame with columns:
    - keyword
    - ruling_favor_pct: % of pro-ruling tweets that are "favor"
    - opp_favor_pct: % of pro-opposition tweets that are "favor"
    - favor_diff: ruling_favor_pct - opp_favor_pct
    - ruling_count: total pro-ruling tweets for this keyword
    - opp_count: total pro-opposition tweets for this keyword
    """
    # Calculate normalized stance percentages
    stats = df.groupby(['keyword', 'party'])['stance'].value_counts(normalize=True).unstack(fill_value=0) * 100
    counts = df.groupby(['keyword', 'party']).size().unstack(fill_value=0)
    
    # Filter to keywords with sufficient data in both parties
    valid_kws = counts[(counts['pro ruling'] > 15) & (counts['pro opposition'] > 15)].index
    
    if valid_kws.empty:
        return None
    
    # Get favor percentages
    stats_valid = stats.loc[valid_kws].reset_index()
    ruling = stats_valid[stats_valid['party'] == 'pro ruling'].set_index('keyword')
    opp = stats_valid[stats_valid['party'] == 'pro opposition'].set_index('keyword')
    
    # Combine into a single DataFrame
    result = pd.DataFrame({
        'keyword': valid_kws,
        'ruling_favor_pct': ruling.loc[valid_kws, 'favor'].values if 'favor' in ruling.columns else 0,
        'opp_favor_pct': opp.loc[valid_kws, 'favor'].values if 'favor' in opp.columns else 0,
        'ruling_count': counts.loc[valid_kws, 'pro ruling'].values,
        'opp_count': counts.loc[valid_kws, 'pro opposition'].values
    })
    result['favor_diff'] = result['ruling_favor_pct'] - result['opp_favor_pct']
    
    return result


def plot_single_butterfly(ax, data, title, party_type):
    """
    Plot a single butterfly chart.
    
    party_type: 'pro_ruling' or 'pro_opposition'
    """
    data = data.sort_values('favor_diff')
    y_pos = np.arange(len(data))
    bar_height = 0.7
    
    # Plot bars
    ax.barh(y_pos, -data['opp_favor_pct'], 
            color=PARTY_COLORS['pro opposition'], 
            label='Opposition Favor %', 
            height=bar_height)
    ax.barh(y_pos, data['ruling_favor_pct'], 
            color=PARTY_COLORS['pro ruling'], 
            label='Ruling Favor %', 
            height=bar_height)
    
    # Configure axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(data['keyword'], fontsize=FS_TICK)
    ax.axvline(0, color='black', lw=1)
    
    # Title and labels
    ax.set_title(title, fontsize=FS_TITLE, fontweight='bold', pad=10)
    ax.set_xlabel("← Opposition % Favor | Ruling % Favor →", fontsize=FS_LABEL)
    ax.tick_params(axis='x', labelsize=FS_TICK)
    ax.legend(fontsize=FS_TICK - 1, loc='lower right')


def plot_split_butterfly_charts(df, output_dir, neutral_threshold=NEUTRAL_THRESHOLD):
    """
    Create two butterfly charts:
    1. Aspects favored by Pro-Ruling (positive difference)
    2. Aspects favored by Pro-Opposition (negative difference)
    
    Neutral aspects (|difference| < threshold) are excluded.
    """
    print("\nGenerating Split Butterfly Favor charts...")
    print(f"  Neutral threshold: ±{neutral_threshold}% points")
    
    # Calculate stats
    stats = calculate_favor_stats(df)
    if stats is None:
        print("  No valid keywords found!")
        return
    
    print(f"  Total keywords with sufficient data: {len(stats)}")
    
    # Split by favor direction, excluding neutral
    pro_ruling_favored = stats[stats['favor_diff'] > neutral_threshold].copy()
    pro_opp_favored = stats[stats['favor_diff'] < -neutral_threshold].copy()
    neutral = stats[abs(stats['favor_diff']) <= neutral_threshold]
    
    print(f"  Pro-Ruling favored aspects: {len(pro_ruling_favored)}")
    print(f"  Pro-Opposition favored aspects: {len(pro_opp_favored)}")
    print(f"  Neutral aspects (excluded): {len(neutral)}")
    
    if len(neutral) > 0:
        print(f"    Excluded: {', '.join(neutral['keyword'].tolist())}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Pro-Ruling favored aspects
    if len(pro_ruling_favored) > 0:
        fig_height = max(4, len(pro_ruling_favored) * 0.4 + 2)
        fig, ax = plt.subplots(figsize=(12, fig_height))
        
        plot_single_butterfly(
            ax, pro_ruling_favored,
            "Aspects Where Pro-Ruling Shows Higher Favor %",
            'pro_ruling'
        )
        
        # Add note about normalization
        fig.text(0.5, 0.02, 
                 "Normalized: Favor % = (favor tweets / total tweets) × 100 for each party-keyword pair",
                 ha='center', fontsize=8, style='italic', alpha=0.7)
        
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        output_path = output_dir / 'butterfly_pro_ruling_favored.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")
    else:
        print("  No Pro-Ruling favored aspects to plot!")
    
    # Plot 2: Pro-Opposition favored aspects
    if len(pro_opp_favored) > 0:
        fig_height = max(4, len(pro_opp_favored) * 0.4 + 2)
        fig, ax = plt.subplots(figsize=(12, fig_height))
        
        plot_single_butterfly(
            ax, pro_opp_favored,
            "Aspects Where Pro-Opposition Shows Higher Favor %",
            'pro_opposition'
        )
        
        # Add note about normalization
        fig.text(0.5, 0.02, 
                 "Normalized: Favor % = (favor tweets / total tweets) × 100 for each party-keyword pair",
                 ha='center', fontsize=8, style='italic', alpha=0.7)
        
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        output_path = output_dir / 'butterfly_pro_opposition_favored.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")
    else:
        print("  No Pro-Opposition favored aspects to plot!")
    
    # Also save stats to CSV for reference
    stats_path = output_dir / 'favor_stats_by_keyword.csv'
    stats['category'] = stats['favor_diff'].apply(
        lambda x: 'pro_ruling_favored' if x > neutral_threshold 
                  else ('pro_opposition_favored' if x < -neutral_threshold else 'neutral')
    )
    stats.to_csv(stats_path, index=False)
    print(f"  Saved stats: {stats_path}")


def main():
    """Main function to generate the visualizations."""
    print("=" * 60)
    print("SPLIT BUTTERFLY FAVOR NORMALIZED VISUALIZATIONS")
    print("=" * 60)
    
    # Set output directory
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput: {output_dir}\n")
    
    # Load and preprocess
    df = load_data()
    
    # Generate visualizations
    plot_split_butterfly_charts(df, output_dir)
    
    print("\n" + "=" * 60)
    print("COMPLETE! Visualizations saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()
