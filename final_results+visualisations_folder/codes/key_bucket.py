#!/usr/bin/env python3
"""
Party Focus with Stance Breakdown Visualization

This script creates a visualization showing party focus and stance breakdown
for different topic buckets. Based on the last cell of stance_advanced_visualisation_part2.ipynb.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import os

warnings.filterwarnings('ignore')

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Color schemes
STANCE_COLORS = {
    'favor': '#2ecc71',
    'against': '#e74c3c',
    'neutral': '#95a5a6'
}

PARTY_COLORS = {
    'pro ruling': '#FF6B35',
    'pro opposition': '#004E89'
}

BUCKET_COLORS = {
    'Economic': '#1abc9c',
    'Social/Religious': '#9b59b6',
    'Political/Policy': '#3498db',
    'Leadership': '#e67e22'
}

KEYWORD_BUCKETS = {
    "Leader & Party Contestation": [
        "modi",
        "rahulgandhi",
        "congress",
    ],

    "Institutions, Democracy & State Accountability": [
        "democracy",
        "dictatorship",
        "spyware",
        "new parliament",
    ],

    "Economy, Development & Macro-Stewardship": [
        "aatmanirbhar",
        "demonetisation",
        "gdp",
        "inflation",
        "unemployment",
        "suicides",
    ],

    "Agrarian Reform & Farmer Movement": [
        "farm laws",
        "farmers protests",
        "msp",
    ],

    "Citizenship, Belonging & Mass Protest Politics": [
        "caa",
        "shaheen bagh",
    ],

    "Majoritarian Ideology & Hindu Nationalist Mobilization": [
        "hindutva",
        "sangh",
        "bhakts",
        "hindu",
    ],

    "Communal Relations, Minority Rights & Collective Violence": [
        "minorities",
        "muslim",
        "lynching",
        "sharia",
        "islamists",
        "hathras",
    ],

    "Symbolic Nationhood & Cultural-Religious Projects": [
        "ayodhya",
        "ram mandir",
        "mahotsav",
    ],

    "Security, Territory & Geopolitics": [
        "china",
        "kashmir",
        "balochistan",
        "kashmiri pandits",
    ],
}



def get_bucket(keyword):
    """Map a keyword to its corresponding bucket category."""
    keyword_lower = keyword.lower()
    for bucket, keywords in KEYWORD_BUCKETS.items():
        for kw in keywords:
            if kw in keyword_lower:
                return bucket
    return 'Other'


def load_data():
    """Load and preprocess the stance data."""
    # Path to the CSV file
    csv_path = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/final_results+visualisations_folder/combined_stance_results.csv"
    
    # Try alternative paths if the main one doesn't exist
    if not Path(csv_path).exists():
        csv_path = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/6_stance/results/final_en_results/stance_results_37keywords.csv"
    
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df):,} rows from {Path(csv_path).name}")
    
    # Filter and standardize
    valid_stances = ['favor', 'against', 'neutral']
    df = df[df['fewshot_label'].isin(valid_stances)].copy()
    df['stance'] = df['fewshot_label']
    df['party'] = df['_label_norm'].str.lower().str.strip()
    df['keyword'] = df['keyword'].str.lower().str.strip()
    df = df[df['party'].isin(['pro ruling', 'pro opposition'])]
    
    # Add bucket classification
    df['bucket'] = df['keyword'].apply(get_bucket)
    
    print(f"\n✓ Final dataset: {len(df):,} tweets")
    return df


def create_party_focus_visualization(df, output_path=None):
    """
    Create the Party Focus with Stance Breakdown visualization.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: party, bucket, stance (with 'favor', 'neutral', 'against')
    output_path : str, optional
        Path to save the output file. If None, saves to default location.
    """
    # 1. Calculate the percentage of total party tweets for each bucket-stance pair
    party_totals = df.groupby('party').size()
    focus_stance = df.groupby(['party', 'bucket', 'stance']).size().unstack(fill_value=0)
    
    # Normalize by party totals to get % of total party discourse
    focus_stance_pct = focus_stance.div(party_totals, axis=0) * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 10), sharey=False)
    
    for idx, party in enumerate(['pro ruling', 'pro opposition']):
        # Filter data for the specific party and sort by total bucket volume
        data = focus_stance_pct.loc[party].copy()
        data['total'] = data.sum(axis=1)
        data = data.sort_values('total', ascending=True)
        data = data.drop(columns=['total'])
        
        # Ensure we have all three stance columns (favor, neutral, against)
        for stance in ['favor', 'neutral', 'against']:
            if stance not in data.columns:
                data[stance] = 0
        
        # Plot stacked bar
        data[['favor', 'neutral', 'against']].plot(
            kind='barh', 
            stacked=True, 
            ax=axes[idx],
            color=[STANCE_COLORS['favor'], STANCE_COLORS['neutral'], STANCE_COLORS['against']],
            edgecolor='white',
            width=0.8
        )
        
        # Styling
        axes[idx].set_title(f'{party.upper()} Focus & Stance', fontweight='bold', fontsize=15, pad=20)
        axes[idx].set_xlabel('Percentage of Total Party Tweets (%)', fontweight='bold')
        axes[idx].set_ylabel('Topic Bucket', fontweight='bold')
        axes[idx].grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add total volume labels at the end of each bar
        for i, (bucket, row) in enumerate(data.iterrows()):
            total_pct = row.sum()
            axes[idx].text(total_pct + 0.5, i, f'{total_pct:.1f}%', va='center', fontweight='bold', fontsize=10)
    
    # Single legend for the whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, [l.capitalize() for l in labels], loc='upper center', 
               bbox_to_anchor=(0.5, 1.05), ncol=3, fontsize=12, title="Stance")
    
    # Remove individual legends
    axes[0].get_legend().remove()
    axes[1].get_legend().remove()
    
    plt.tight_layout()
    
    # Set output path
    if output_path is None:
        output_dir = Path(__file__).parent / 'output'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'party_focus_stance_breakdown.png'
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    
    plt.show()
    return fig


def main():
    """Main function to run the visualization."""
    print("=" * 60)
    print("PARTY FOCUS WITH STANCE BREAKDOWN")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Create visualization
    create_party_focus_visualization(df)
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()