#!/usr/bin/env python3
"""
Combined Hindi-English-Hinglish Stacked Stance Plot
===================================================
Creates a publication-quality stacked bar chart showing stance distribution
for English, Hindi, and Hinglish tweets across multiple keywords using the
final merged stance CSV in this folder.

Layout:
- 3 rows x 6 columns (3 keywords per row, 6 bars per keyword for the 3 languages)
- Row 1: Hindutva, Modi, Ram Mandir
- Row 2: Democracy, Kashmir, Minorities  
- Row 3: Muslim, Rahul Gandhi, Farmers Protests
- Row labels: Highly Ruling Favors, Neutral, Opposition Favors

Each subplot shows:
- 6 bars: Pro Ruling English/Hindi/Hinglish and Pro Opposition English/Hindi/Hinglish
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

BASE_DIR = Path(__file__).parent.resolve()
DATA_CSV = BASE_DIR / "extracted_ALL_35keywords_english_hindi_only_final_with_existing_and_inferred_stance.csv"
OUTPUT_DIR = BASE_DIR

# Keywords organized by row
KEYWORDS_BY_ROW = [
    ['Hindutva', 'Modi', 'Ram Mandir'],
    ['Democracy', 'Kashmir', 'Minorities'],
    ['Muslim', 'Rahulgandhi', 'Farmers Protests']
]

ROW_LABELS = ['Highly Pro Ruling Favors', 'Neutral', 'Highly Pro Opposition Favours']

# Publication-quality styling
plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.weight': 'normal',  # No bold per LaTeX guidelines
    'axes.edgecolor': '#333333',
    'axes.labelweight': 'normal',
    'axes.titleweight': 'normal',  # No bold titles
    'xtick.labelsize': 12,
    'ytick.labelsize': 11,
    'legend.fontsize': 18,
    'figure.titlesize': 16
})

# Override with larger font sizes for 12-panel plot readability
plt.rcParams.update({
    'font.size': 18,          # global default
    'axes.titlesize': 26,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 16,
    'legend.fontsize': 20,
})

# Color scheme for stances
PUB_COLORS = {
    'favor':   '#27ae60',  # Emerald Green
    'neutral': '#95a5a6',  # Slate Grey
    'against': '#c0392b'   # Crimson Red
}

# Language differentiation via edge colors (cleaner than hatching)
LANGUAGE_EDGE = {
    'english': {'color': 'white', 'linewidth': 1.5},
    'hindi': {'color': '#2c3e50', 'linewidth': 2.5},
    'hinglish': {'color': '#7d6608', 'linewidth': 2.5},
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
    df['translation_route'] = df['translation_route'].fillna('').astype(str).str.lower().str.strip()
    df['lid_lang'] = df['lid_lang'].fillna('').astype(str).str.lower().str.strip()
    df['language_group'] = df['language']
    hinglish_mask = (df['translation_route'] == 'hinglish_roman') | (df['lid_lang'] == 'hin_latn')
    df.loc[hinglish_mask, 'language_group'] = 'hinglish'
    
    # Filter to only pro ruling and pro opposition
    df = df[df['party'].isin(['pro ruling', 'pro opposition'])]
    
    # Filter to only English, Hindi, and Hinglish.
    df = df[df['language_group'].isin(['english', 'hindi', 'hinglish'])]
    
    print(f"Loaded {len(df):,} total tweets")
    print(f"  English: {len(df[df['language_group']=='english']):,}")
    print(f"  Hindi: {len(df[df['language_group']=='hindi']):,}")
    print(f"  Hinglish: {len(df[df['language_group']=='hinglish']):,}")
    
    return df

# =============================================================================
# PLOTTING FUNCTION
# =============================================================================

def create_combined_stance_plot(df):
    """Create the combined English-Hindi-Hinglish stacked stance plot."""
    print("\nGenerating combined stance plot...")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Flatten keywords for iteration
    all_keywords = [kw for row in KEYWORDS_BY_ROW for kw in row]
    n_rows = len(KEYWORDS_BY_ROW)
    n_cols = len(KEYWORDS_BY_ROW[0])
    
    # Create figure - larger for better readability
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(38, 26), facecolor='white')
    axes = axes.flatten()
    
    # Define bar groups: Party + Language combinations
    bar_groups = [
        ('pro ruling', 'english', 'Ruling\nEnglish'),
        ('pro ruling', 'hindi', 'Ruling\nHindi'),
        ('pro ruling', 'hinglish', 'Ruling\nHinglish'),
        ('pro opposition', 'english', 'Opp.\nEnglish'),
        ('pro opposition', 'hindi', 'Opp.\nHindi'),
        ('pro opposition', 'hinglish', 'Opp.\nHinglish'),
    ]
    
    # Plotting loop
    for i, kw in enumerate(all_keywords):
        ax = axes[i]
        kw_lower = kw.lower()
        kw_data = df[df['keyword'] == kw_lower]
        
        if kw_data.empty:
            ax.set_title(f'{kw}\n(No Data)', fontsize=13)
            ax.set_axis_off()
            continue
        
        # Prepare data for each bar group
        bar_data = []
        bar_counts = []
        x_labels = []
        
        for party, lang, label in bar_groups:
            subset = kw_data[(kw_data['party'] == party) & (kw_data['language_group'] == lang)]
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
                                 color='black', fontsize=24, fontweight='normal',
                                 fontfamily='serif',
                                 linespacing=1.0)
        
        # Clean up axes
        clean_title = kw.replace("Rahulgandhi", "Rahul Gandhi").title()
        # Title removed for LaTeX (caption will be in LaTeX document)
        # Just show keyword name as axis label
        ax.set_title(f'{clean_title}', fontsize=38, pad=18)
        
        ax.set_ylim(0, 100)
        ax.set_xlim(-0.5, len(bar_data) - 0.5)
        ax.set_xlabel('')
        
        # Y-axis label only on leftmost columns
        if i % n_cols == 0:
            ax.set_ylabel('Percentage of Tweets (%)', fontsize=24)
        else:
            ax.set_ylabel('')
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels([d['label'] for d in bar_data], rotation=0, fontsize=18)
        
        # Tick font sizes
        ax.tick_params(axis='y', labelsize=20)
        ax.tick_params(axis='x', labelsize=18)
        
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
        Patch(facecolor='#7f8c8d', edgecolor='#2c3e50', linewidth=3, label='Hindi'),
        Patch(facecolor='#7f8c8d', edgecolor='#7d6608', linewidth=3, label='Hinglish'),
    ]
    
    # Combined legend at top - LARGER
    all_legend = stance_legend + lang_legend
    fig.legend(handles=all_legend, loc='upper center', bbox_to_anchor=(0.5, 1.01),
               ncol=6, fontsize=22, frameon=True, fancybox=True,
               edgecolor='#cccccc', facecolor='white',
               prop={'family': 'serif'})
    
    # ==========================================================================
    # FINAL ADJUSTMENTS
    # ==========================================================================
    
    # Don't use tight_layout() - it shrinks fonts when combined with bbox_inches='tight'
    # Reduced margins to minimize whitespace
    plt.subplots_adjust(top=0.94, bottom=0.04, left=0.06, right=0.98, hspace=0.35, wspace=0.12)
    
    # Row labels (bold) aligned to row centers
    if len(ROW_LABELS) == n_rows:
        for row_idx, row_label in enumerate(ROW_LABELS):
            left_ax = axes[row_idx * n_cols]
            bbox = left_ax.get_position()
            y_center = (bbox.y0 + bbox.y1) / 2
            x_pos = max(0.02, bbox.x0 - 0.03)
            fig.text(x_pos, y_center, row_label,
                     ha='center', va='center',
                     rotation=90,
                     fontsize=16,
                     family='serif')
    else:
        warnings.warn("ROW_LABELS length does not match number of rows.")
    
    # Save both PNG and PDF in the FINAL_STANCE folder.
    save_path_png = OUTPUT_DIR / 'combined_hindi_english_hinglish_stance_stacked.png'
    save_path_pdf = OUTPUT_DIR / 'combined_hindi_english_hinglish_stance_stacked.pdf'
    plt.savefig(save_path_png, dpi=300, facecolor='white', bbox_inches='tight')
    plt.savefig(save_path_pdf, dpi=300, facecolor='white', format='pdf')
    plt.close()
    
    print(f"\nPlot saved to: {save_path_png}")
    print(f"Plot saved to: {save_path_pdf}")
    return save_path_png


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main function."""
    print("=" * 70)
    print("COMBINED HINDI-ENGLISH-HINGLISH STANCE STACKED PLOT")
    print("=" * 70)
    
    # Load data
    df = load_data()
    
    # Create plot
    save_path = create_combined_stance_plot(df)
    
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
