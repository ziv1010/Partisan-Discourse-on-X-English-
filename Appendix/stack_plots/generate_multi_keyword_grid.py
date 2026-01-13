#!/usr/bin/env python3
"""
Multi-Keyword Grid Stacked Stance Plots
========================================
Creates a grid layout with multiple keywords in one figure.
Configure KEYWORDS_GRID to specify which keywords appear together.
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
# CONFIGURATION - MODIFY THIS SECTION
# =============================================================================

DATA_CSV = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/final_results+visualisations_folder/combined_stance_results.csv"
OUTPUT_DIR = Path("/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Appendix/stack_plots")

# Configure your grid layouts here. Each entry creates one output file.
# Format: {'output_name': [list of keywords]}
# The grid will automatically arrange based on number of keywords (up to 3 cols)
KEYWORD_GRIDS = {
    'religious_themes': ['hindutva', 'muslim', 'ram mandir', 'ayodhya', 'minorities', 'islamists'],
    'political_figures': ['modi', 'rahulgandhi', 'congress'],
    'economic_issues': ['demonetisation', 'gdp', 'inflation', 'unemployment', 'msp'],
    'farmer_issues': ['farmers protests', 'farm laws', 'msp', 'suicides'],
    'governance': ['democracy', 'dictatorship', 'caa', 'ucc', 'new parliament'],
    'regional_issues': ['kashmir', 'kashmiri pandits', 'balochistan', 'china'],
    'social_issues': ['hathras', 'lynching', 'shaheen bagh', 'sharia'],
    'all_keywords': None  # Set to None to include ALL keywords
}

# =============================================================================
# STYLING
# =============================================================================

plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'axes.edgecolor': '#333333'
})

PUB_COLORS = {
    'favor':   '#27ae60',
    'neutral': '#95a5a6',
    'against': '#c0392b'
}

LANGUAGE_EDGE = {
    'english': {'color': 'white', 'linewidth': 1.5},
    'hindi': {'color': '#2c3e50', 'linewidth': 2.5}
}

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load and preprocess the combined stance results."""
    print("Loading data...")
    df = pd.read_csv(DATA_CSV, low_memory=False)
    
    valid_stances = ['favor', 'against', 'neutral']
    df = df[df['fewshot_label'].isin(valid_stances)].copy()
    
    df['stance'] = df['fewshot_label'].str.lower().str.strip()
    df['party'] = df['_label_norm'].str.lower().str.strip()
    df['keyword'] = df['keyword'].str.lower().str.strip()
    df['language'] = df['language'].str.lower().str.strip()
    
    df = df[df['party'].isin(['pro ruling', 'pro opposition'])]
    df = df[df['language'].isin(['english', 'hindi'])]
    
    print(f"Loaded {len(df):,} total tweets")
    return df

# =============================================================================
# PLOTTING
# =============================================================================

def create_grid_plot(df, keywords, output_name, output_dir):
    """Create a grid layout with multiple keywords."""
    
    n_keywords = len(keywords)
    if n_keywords == 0:
        return None
    
    # Determine grid dimensions
    n_cols = min(3, n_keywords)
    n_rows = (n_keywords + n_cols - 1) // n_cols
    
    # Figure size scales with grid
    fig_width = 8 * n_cols
    fig_height = 6 * n_rows
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), facecolor='white')
    
    # Handle single row/col cases
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    axes_flat = axes.flatten()
    
    bar_groups = [
        ('pro ruling', 'english', 'Ruling\nEnglish'),
        ('pro ruling', 'hindi', 'Ruling\nHindi'),
        ('pro opposition', 'english', 'Opp.\nEnglish'),
        ('pro opposition', 'hindi', 'Opp.\nHindi')
    ]
    
    for idx, kw in enumerate(keywords):
        ax = axes_flat[idx]
        kw_lower = kw.lower()
        kw_data = df[df['keyword'] == kw_lower]
        
        if kw_data.empty:
            ax.set_title(f'{kw}\n(No Data)', fontsize=13, fontweight='bold')
            ax.set_axis_off()
            continue
        
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
                'label': label, 'party': party, 'language': lang,
                'favor': pct_favor, 'neutral': pct_neutral, 'against': pct_against,
                'n': n, 'count_favor': count_favor, 
                'count_neutral': count_neutral, 'count_against': count_against
            })
        
        x_positions = np.arange(len(bar_data))
        bar_width = 0.65
        
        bottom_favor = np.zeros(len(bar_data))
        bottom_neutral = np.array([d['favor'] for d in bar_data])
        bottom_against = np.array([d['favor'] + d['neutral'] for d in bar_data])
        
        for stance, color, bottoms in [
            ('favor', PUB_COLORS['favor'], bottom_favor),
            ('neutral', PUB_COLORS['neutral'], bottom_neutral),
            ('against', PUB_COLORS['against'], bottom_against)
        ]:
            heights = np.array([d[stance] for d in bar_data])
            
            for j, (bar_d, h, b) in enumerate(zip(bar_data, heights, bottoms)):
                edge_style = LANGUAGE_EDGE.get(bar_d['language'], {'color': 'white', 'linewidth': 1.5})
                ax.bar(x_positions[j], h, bar_width, bottom=b, 
                       color=color, edgecolor=edge_style['color'], linewidth=edge_style['linewidth'])
                
                if h > 10:
                    count = bar_d[f'count_{stance}']
                    txt = ax.text(x_positions[j], b + h/2, f'{h:.0f}%\n(n={count:,})',
                                 ha='center', va='center', color='white', fontsize=10)
                    txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground=(0,0,0,0.5))])
        
        clean_title = kw.replace("rahulgandhi", "Rahul Gandhi").replace("_", " ").title()
        ax.set_title(f'{clean_title}\nN = {len(kw_data):,}', fontsize=14, fontweight='bold', pad=10)
        ax.set_ylim(0, 100)
        ax.set_xlim(-0.5, len(bar_data) - 0.5)
        ax.set_ylabel('Percentage (%)' if idx % n_cols == 0 else '', fontsize=11)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([d['label'] for d in bar_data], fontsize=10, fontweight='bold')
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Hide unused axes
    for idx in range(len(keywords), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    # Legend
    legend_handles = [
        Patch(facecolor=PUB_COLORS['favor'], edgecolor='white', label='Favor'),
        Patch(facecolor=PUB_COLORS['neutral'], edgecolor='white', label='Neutral'),
        Patch(facecolor=PUB_COLORS['against'], edgecolor='white', label='Against'),
        Patch(facecolor='#7f8c8d', edgecolor='white', linewidth=2, label='English'),
        Patch(facecolor='#7f8c8d', edgecolor='#2c3e50', linewidth=3, label='Hindi')
    ]
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.01),
               ncol=5, fontsize=14, frameon=True, fancybox=True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.35, wspace=0.15)
    
    save_path = output_dir / f'grid_{output_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return save_path

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("GENERATING MULTI-KEYWORD GRID PLOTS")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    all_keywords = sorted(df['keyword'].dropna().unique().tolist())
    
    print(f"\nGenerating {len(KEYWORD_GRIDS)} grid plots...")
    
    for grid_name, keywords in KEYWORD_GRIDS.items():
        if keywords is None:
            keywords = all_keywords
        
        save_path = create_grid_plot(df, keywords, grid_name, OUTPUT_DIR)
        if save_path:
            print(f"  ✓ {grid_name}: {len(keywords)} keywords -> {save_path.name}")
    
    print("\n" + "=" * 70)
    print(f"COMPLETE! Output: {OUTPUT_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
