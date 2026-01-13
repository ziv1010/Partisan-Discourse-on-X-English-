#!/usr/bin/env python3
"""
Individual Keyword Stacked Stance Plots
========================================
Generates a separate stacked bar chart for each keyword in the dataset,
showing stance distribution for English and Hindi tweets.

Each plot shows:
- 4 bars: Pro Ruling English, Pro Ruling Hindi, Pro Opposition English, Pro Opposition Hindi
- Stacked by stance: Favor, Neutral, Against
- Percentage labels with counts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Patch
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_CSV = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/final_results+visualisations_folder/combined_stance_results.csv"
OUTPUT_DIR = Path("/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Appendix/stack_plots")

# Publication-quality styling
plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'axes.edgecolor': '#333333'
})

# Color scheme for stances
PUB_COLORS = {
    'favor':   '#27ae60',  # Emerald Green
    'neutral': '#95a5a6',  # Slate Grey
    'against': '#c0392b'   # Crimson Red
}

# Language differentiation via edge colors
LANGUAGE_EDGE = {
    'english': {'color': 'white', 'linewidth': 1.5},
    'hindi': {'color': '#2c3e50', 'linewidth': 2.5}  # Dark outline for Hindi
}

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_data():
    """Load and preprocess the combined stance results."""
    print("Loading data...")
    df = pd.read_csv(DATA_CSV, low_memory=False)
    
    # Filter to valid stances
    valid_stances = ['favor', 'against', 'neutral']
    df = df[df['fewshot_label'].isin(valid_stances)].copy()
    
    # Standardize columns
    df['stance'] = df['fewshot_label'].str.lower().str.strip()
    df['party'] = df['_label_norm'].str.lower().str.strip()
    df['keyword'] = df['keyword'].str.lower().str.strip()
    df['language'] = df['language'].str.lower().str.strip()
    
    # Filter to only pro ruling and pro opposition
    df = df[df['party'].isin(['pro ruling', 'pro opposition'])]
    
    # Filter to only English and Hindi
    df = df[df['language'].isin(['english', 'hindi'])]
    
    print(f"Loaded {len(df):,} total tweets")
    print(f"  English: {len(df[df['language']=='english']):,}")
    print(f"  Hindi: {len(df[df['language']=='hindi']):,}")
    
    return df

# =============================================================================
# PLOTTING FUNCTION FOR A SINGLE KEYWORD
# =============================================================================

def create_single_keyword_plot(df, keyword, output_dir):
    """Create a stacked stance plot for a single keyword."""
    
    kw_lower = keyword.lower()
    kw_data = df[df['keyword'] == kw_lower]
    
    if kw_data.empty:
        print(f"  Skipping '{keyword}' - No data")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    
    # Define bar groups: Party + Language combinations
    bar_groups = [
        ('pro ruling', 'english', 'Ruling\nEnglish'),
        ('pro ruling', 'hindi', 'Ruling\nHindi'),
        ('pro opposition', 'english', 'Opp.\nEnglish'),
        ('pro opposition', 'hindi', 'Opp.\nHindi')
    ]
    
    # Prepare data for each bar group
    bar_data = []
    
    for party, lang, label in bar_groups:
        subset = kw_data[(kw_data['party'] == party) & (kw_data['language'] == lang)]
        n = len(subset)
        
        if n > 0:
            pct_favor = (subset['stance'] == 'favor').sum() / n * 100
            pct_neutral = (subset['stance'] == 'neutral').sum() / n * 100
            pct_against = (subset['stance'] == 'against').sum() / n * 100
            
            count_favor = (subset['stance'] == 'favor').sum()
            count_neutral = (subset['stance'] == 'neutral').sum()
            count_against = (subset['stance'] == 'against').sum()
        else:
            pct_favor = pct_neutral = pct_against = 0
            count_favor = count_neutral = count_against = 0
        
        bar_data.append({
            'label': label,
            'party': party,
            'language': lang,
            'favor': pct_favor,
            'neutral': pct_neutral,
            'against': pct_against,
            'n': n,
            'count_favor': count_favor,
            'count_neutral': count_neutral,
            'count_against': count_against
        })
    
    # Plot stacked bars
    x_positions = np.arange(len(bar_data))
    bar_width = 0.65
    
    # Bottom positions for stacking
    bottom_favor = np.zeros(len(bar_data))
    bottom_neutral = np.array([d['favor'] for d in bar_data])
    bottom_against = np.array([d['favor'] + d['neutral'] for d in bar_data])
    
    # Draw bars for each stance
    for stance, color, bottoms in [
        ('favor', PUB_COLORS['favor'], bottom_favor),
        ('neutral', PUB_COLORS['neutral'], bottom_neutral),
        ('against', PUB_COLORS['against'], bottom_against)
    ]:
        heights = np.array([d[stance] for d in bar_data])
        
        for j, (bar_d, h, b) in enumerate(zip(bar_data, heights, bottoms)):
            edge_style = LANGUAGE_EDGE.get(bar_d['language'], {'color': 'white', 'linewidth': 1.5})
            bar = ax.bar(x_positions[j], h, bar_width, bottom=b, 
                        color=color, 
                        edgecolor=edge_style['color'], 
                        linewidth=edge_style['linewidth'])
            
            # Add percentage + count labels for segments > 10%
            if h > 10:
                count_key = f'count_{stance}'
                count = bar_d[count_key]
                
                x_pos = x_positions[j]
                y_pos = b + h / 2
                label_text = f'{h:.0f}%\n(n={count:,})'
                
                txt = ax.text(x_pos, y_pos, label_text,
                             ha='center', va='center',
                             color='white', fontsize=12,
                             linespacing=1.0)
                txt.set_path_effects([
                    path_effects.withStroke(linewidth=3, foreground=(0, 0, 0, 0.5))
                ])
    
    # Clean up axes
    clean_title = keyword.replace("rahulgandhi", "Rahul Gandhi").replace("_", " ").title()
    total_n = len(kw_data)
    ax.set_title(f'{clean_title}\nTotal N = {total_n:,}', 
                 fontsize=18, fontweight='bold', pad=16)
    
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.5, len(bar_data) - 0.5)
    ax.set_xlabel('')
    ax.set_ylabel('Percentage of Tweets (%)', fontsize=14)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels([d['label'] for d in bar_data], 
                      rotation=0, fontsize=12, fontweight='bold')
    
    ax.tick_params(axis='y', labelsize=11)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='#bdc3c7')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # ==========================================================================
    # LEGEND
    # ==========================================================================
    
    # Stance legend
    stance_legend = [
        Patch(facecolor=PUB_COLORS['favor'], edgecolor='white', label='Favor'),
        Patch(facecolor=PUB_COLORS['neutral'], edgecolor='white', label='Neutral'),
        Patch(facecolor=PUB_COLORS['against'], edgecolor='white', label='Against')
    ]
    
    # Language legend (using edge colors)
    lang_legend = [
        Patch(facecolor='#7f8c8d', edgecolor='white', linewidth=2, label='English'),
        Patch(facecolor='#7f8c8d', edgecolor='#2c3e50', linewidth=3, label='Hindi (Dark Edge)')
    ]
    
    # Combined legend at top
    all_legend = stance_legend + lang_legend
    ax.legend(handles=all_legend, loc='upper center', bbox_to_anchor=(0.5, -0.08),
              ncol=5, fontsize=12, frameon=True, fancybox=True,
              edgecolor='#cccccc', facecolor='white',
              prop={'family': 'serif', 'weight': 'bold'})
    
    # ==========================================================================
    # SAVE
    # ==========================================================================
    
    plt.tight_layout()
    
    # Sanitize filename
    safe_keyword = keyword.lower().replace(" ", "_").replace("/", "_")
    save_path = output_dir / f'stance_stacked_{safe_keyword}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return save_path

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main function."""
    print("=" * 70)
    print("GENERATING INDIVIDUAL KEYWORD STACKED STANCE PLOTS")
    print("=" * 70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_data()
    
    # Get all unique keywords
    all_keywords = df['keyword'].dropna().unique()
    print(f"\nFound {len(all_keywords)} unique keywords")
    
    # Generate plot for each keyword
    print("\nGenerating individual plots...")
    generated_count = 0
    
    for keyword in sorted(all_keywords):
        save_path = create_single_keyword_plot(df, keyword, OUTPUT_DIR)
        if save_path:
            print(f"  ✓ {keyword}: {save_path.name}")
            generated_count += 1
    
    print("\n" + "=" * 70)
    print(f"COMPLETE! Generated {generated_count} plots")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
