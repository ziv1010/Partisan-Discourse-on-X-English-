#!/usr/bin/env python3
"""
Appendix Table Generator v2: Aspect Contribution to Bucket Stance

For each bucket, shows how each aspect CONTRIBUTES to the bucket's overall stance percentages.
E.g., if a bucket has 54% Favor, shows: Modi=30%, Rahul=15%, Congress=9% → totals to 54%
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

KEYWORD_BUCKETS = {
    "Leader & Party Contestation": [
        "modi", "rahulgandhi", "congress",
    ],
    "Institutions, Democracy & State Accountability": [
        "democracy", "dictatorship", "spyware", "new parliament",
    ],
    "Economy, Development & Macro-Stewardship": [
        "aatmanirbhar", "demonetisation", "gdp", "inflation", "unemployment", "suicides",
    ],
    "Agrarian Reform & Farmer Movement": [
        "farm laws", "farmers protests", "msp",
    ],
    "Citizenship, Belonging & Mass Protest Politics": [
        "caa", "shaheen bagh",
    ],
    "Majoritarian Ideology & Hindu Nationalist Mobilization": [
        "hindutva", "sangh", "bhakts", "hindu",
    ],
    "Communal Relations, Minority Rights & Collective Violence": [
        "minorities", "muslim", "lynching", "sharia", "islamists", "hathras",
    ],
    "Symbolic Nationhood & Cultural-Religious Projects": [
        "ayodhya", "ram mandir", "mahotsav",
    ],
    "Security, Territory & Geopolitics": [
        "china", "kashmir", "balochistan", "kashmiri pandits",
    ],
}


def get_bucket(keyword):
    keyword_lower = keyword.lower()
    for bucket, keywords in KEYWORD_BUCKETS.items():
        for kw in keywords:
            if kw in keyword_lower:
                return bucket
    return 'Other'


def load_data():
    csv_path = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/final_results+visualisations_folder/combined_stance_results.csv"
    
    if not Path(csv_path).exists():
        csv_path = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/6_stance/results/final_en_results/stance_results_37keywords.csv"
    
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df):,} rows")
    
    valid_stances = ['favor', 'against', 'neutral']
    df = df[df['fewshot_label'].isin(valid_stances)].copy()
    df['stance'] = df['fewshot_label']
    df['party'] = df['_label_norm'].str.lower().str.strip()
    df['keyword'] = df['keyword'].str.lower().str.strip()
    df = df[df['party'].isin(['pro ruling', 'pro opposition'])]
    df['bucket'] = df['keyword'].apply(get_bucket)
    
    print(f"✓ Final dataset: {len(df):,} tweets")
    return df


def generate_contribution_table(df):
    """
    Generate table showing aspect contribution to bucket stance percentages.
    
    For each bucket:
    - Bucket Total = sum of all tweets in bucket
    - Each aspect's contribution % = (aspect's stance count / bucket total) * 100
    - Sum of all aspect contributions for a stance = bucket's overall stance %
    """
    
    rows = []
    
    for bucket, keywords in KEYWORD_BUCKETS.items():
        bucket_df = df[df['bucket'] == bucket]
        bucket_total = len(bucket_df)
        
        if bucket_total == 0:
            continue
        
        # Calculate bucket-level stance percentages
        bucket_favor_total = len(bucket_df[bucket_df['stance'] == 'favor'])
        bucket_against_total = len(bucket_df[bucket_df['stance'] == 'against'])
        bucket_neutral_total = len(bucket_df[bucket_df['stance'] == 'neutral'])
        
        bucket_favor_pct = bucket_favor_total / bucket_total * 100
        bucket_against_pct = bucket_against_total / bucket_total * 100
        bucket_neutral_pct = bucket_neutral_total / bucket_total * 100
        
        for keyword in keywords:
            kw_df = bucket_df[bucket_df['keyword'].str.contains(keyword, case=False, na=False)]
            
            if len(kw_df) == 0:
                continue
            
            # Count by stance
            favor_count = len(kw_df[kw_df['stance'] == 'favor'])
            against_count = len(kw_df[kw_df['stance'] == 'against'])
            neutral_count = len(kw_df[kw_df['stance'] == 'neutral'])
            
            # Calculate contribution to BUCKET total (not aspect total)
            favor_contrib = (favor_count / bucket_total * 100)
            against_contrib = (against_count / bucket_total * 100)
            neutral_contrib = (neutral_count / bucket_total * 100)
            
            rows.append({
                'Bucket': bucket,
                'Aspect': keyword.title(),
                'Favor Contribution (%)': f"{favor_contrib:.1f}%",
                'Favor (n)': favor_count,
                'Against Contribution (%)': f"{against_contrib:.1f}%",
                'Against (n)': against_count,
                'Neutral Contribution (%)': f"{neutral_contrib:.1f}%",
                'Neutral (n)': neutral_count,
                'Aspect Total': len(kw_df),
            })
        
        # Add bucket total row
        rows.append({
            'Bucket': bucket,
            'Aspect': '** BUCKET TOTAL **',
            'Favor Contribution (%)': f"{bucket_favor_pct:.1f}%",
            'Favor (n)': bucket_favor_total,
            'Against Contribution (%)': f"{bucket_against_pct:.1f}%",
            'Against (n)': bucket_against_total,
            'Neutral Contribution (%)': f"{bucket_neutral_pct:.1f}%",
            'Neutral (n)': bucket_neutral_total,
            'Aspect Total': bucket_total,
        })
    
    return pd.DataFrame(rows)


def generate_latex_table(df_table, output_path):
    """Generate LaTeX formatted table."""
    
    latex_str = r"""% Appendix Table: Aspect Contribution to Bucket Stance Breakdown
% Each aspect's contribution % adds up to the bucket's total stance %
\begin{longtable}{|p{4cm}|p{2cm}|r|r|r|r|r|r|r|}
\hline
\textbf{Bucket} & \textbf{Aspect} & \textbf{Favor \%} & \textbf{Favor (n)} & \textbf{Against \%} & \textbf{Against (n)} & \textbf{Neutral \%} & \textbf{Neutral (n)} & \textbf{Total} \\
\hline
\endhead
"""
    
    current_bucket = None
    for _, row in df_table.iterrows():
        bucket = row['Bucket']
        aspect = row['Aspect']
        
        if current_bucket and current_bucket != bucket:
            latex_str += r"\hline" + "\n"
        current_bucket = bucket
        
        if aspect == '** BUCKET TOTAL **':
            latex_str += f"\\textbf{{{bucket}}} & \\textbf{{TOTAL}} & \\textbf{{{row['Favor Contribution (%)']}}} & \\textbf{{{row['Favor (n)']}}} & \\textbf{{{row['Against Contribution (%)']}}} & \\textbf{{{row['Against (n)']}}} & \\textbf{{{row['Neutral Contribution (%)']}}} & \\textbf{{{row['Neutral (n)']}}} & \\textbf{{{row['Aspect Total']}}} \\\\\n"
        else:
            bucket_display = bucket if aspect == df_table[df_table['Bucket'] == bucket].iloc[0]['Aspect'] else ""
            latex_str += f"{bucket_display} & {aspect} & {row['Favor Contribution (%)']} & {row['Favor (n)']} & {row['Against Contribution (%)']} & {row['Against (n)']} & {row['Neutral Contribution (%)']} & {row['Neutral (n)']} & {row['Aspect Total']} \\\\\n"
    
    latex_str += r"""\hline
\end{longtable}
"""
    
    with open(output_path, 'w') as f:
        f.write(latex_str)
    
    print(f"✓ LaTeX table saved to: {output_path}")


def main():
    print("=" * 70)
    print("APPENDIX TABLE v2: ASPECT CONTRIBUTION TO BUCKET STANCE")
    print("=" * 70)
    
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = load_data()
    
    # Generate contribution table
    print("\n--- Generating Contribution Table ---")
    contrib_table = generate_contribution_table(df)
    
    csv_path = output_dir / 'appendix_aspect_contribution_to_bucket.csv'
    contrib_table.to_csv(csv_path, index=False)
    print(f"✓ CSV saved to: {csv_path}")
    
    # Print table
    print("\n" + "=" * 100)
    print("FULL TABLE:")
    print("=" * 100)
    print(contrib_table.to_string(index=False))
    
    # Generate LaTeX
    latex_path = output_dir / 'appendix_aspect_contribution_table.tex'
    generate_latex_table(contrib_table, latex_path)
    
    print("\n" + "=" * 70)
    print("✓ Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
