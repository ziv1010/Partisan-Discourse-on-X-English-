#!/usr/bin/env python3
"""
Generate LaTeX Table for Stance Distribution by Keyword
========================================================
Creates a LaTeX table showing stance percentage splits for Pro-Ruling and 
Pro-Opposition for each keyword, along with tweet counts.
"""

import pandas as pd
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_CSV = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/final_results+visualisations_folder/combined_stance_results.csv"
OUTPUT_DIR = Path("/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Appendix/stack_plots")

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
    
    # Filter to only pro ruling and pro opposition
    df = df[df['party'].isin(['pro ruling', 'pro opposition'])]
    
    print(f"Loaded {len(df):,} total tweets")
    
    return df

# =============================================================================
# GENERATE TABLE DATA
# =============================================================================

def generate_table_data(df):
    """Generate table data for each keyword."""
    
    all_keywords = sorted(df['keyword'].dropna().unique())
    
    table_rows = []
    
    for keyword in all_keywords:
        kw_data = df[df['keyword'] == keyword]
        
        if kw_data.empty:
            continue
        
        # Pro Ruling stats
        pr_data = kw_data[kw_data['party'] == 'pro ruling']
        pr_n = len(pr_data)
        if pr_n > 0:
            pr_favor = (pr_data['stance'] == 'favor').sum() / pr_n * 100
            pr_against = (pr_data['stance'] == 'against').sum() / pr_n * 100
            pr_neutral = (pr_data['stance'] == 'neutral').sum() / pr_n * 100
        else:
            pr_favor = pr_against = pr_neutral = 0
        
        # Pro Opposition stats
        po_data = kw_data[kw_data['party'] == 'pro opposition']
        po_n = len(po_data)
        if po_n > 0:
            po_favor = (po_data['stance'] == 'favor').sum() / po_n * 100
            po_against = (po_data['stance'] == 'against').sum() / po_n * 100
            po_neutral = (po_data['stance'] == 'neutral').sum() / po_n * 100
        else:
            po_favor = po_against = po_neutral = 0
        
        # Format keyword for display
        display_keyword = keyword.replace("rahulgandhi", "Rahul Gandhi").replace("_", " ").title()
        
        table_rows.append({
            'keyword': display_keyword,
            'pr_favor': pr_favor,
            'pr_against': pr_against,
            'pr_neutral': pr_neutral,
            'pr_n': pr_n,
            'po_favor': po_favor,
            'po_against': po_against,
            'po_neutral': po_neutral,
            'po_n': po_n,
            'total_n': pr_n + po_n
        })
    
    return table_rows

# =============================================================================
# GENERATE LATEX TABLE
# =============================================================================

def generate_latex_table(table_rows):
    """Generate LaTeX longtable code."""
    
    latex_lines = []
    
    # Table header
    latex_lines.append(r"{\scriptsize")
    latex_lines.append(r"\begin{longtable}{|l|r|r|r|r|r|r|r|r|r|}")
    latex_lines.append(r"\caption{Stance Distribution by Keyword: Percentage splits for Pro-Ruling and Pro-Opposition} \label{tab:stance_distribution_keyword} \\")
    latex_lines.append(r"\hline")
    latex_lines.append(r"\multirow{2}{*}{\textbf{Keyword}} & \multicolumn{4}{c|}{\textbf{Pro-Ruling}} & \multicolumn{4}{c|}{\textbf{Pro-Opposition}} & \multirow{2}{*}{\textbf{Total}} \\")
    latex_lines.append(r"\cline{2-9}")
    latex_lines.append(r" & \textbf{Fav\%} & \textbf{Ag\%} & \textbf{Neu\%} & \textbf{N} & \textbf{Fav\%} & \textbf{Ag\%} & \textbf{Neu\%} & \textbf{N} & \\")
    latex_lines.append(r"\hline")
    latex_lines.append(r"\endfirsthead")
    
    # Continuation header
    latex_lines.append(r"\multicolumn{10}{c}{\tablename\ \thetable{} -- continued from previous page} \\")
    latex_lines.append(r"\hline")
    latex_lines.append(r"\multirow{2}{*}{\textbf{Keyword}} & \multicolumn{4}{c|}{\textbf{Pro-Ruling}} & \multicolumn{4}{c|}{\textbf{Pro-Opposition}} & \multirow{2}{*}{\textbf{Total}} \\")
    latex_lines.append(r"\cline{2-9}")
    latex_lines.append(r" & \textbf{Fav\%} & \textbf{Ag\%} & \textbf{Neu\%} & \textbf{N} & \textbf{Fav\%} & \textbf{Ag\%} & \textbf{Neu\%} & \textbf{N} & \\")
    latex_lines.append(r"\hline")
    latex_lines.append(r"\endhead")
    
    # Continuation footer
    latex_lines.append(r"\hline \multicolumn{10}{r}{{Continued on next page}} \\")
    latex_lines.append(r"\endfoot")
    latex_lines.append(r"\hline")
    latex_lines.append(r"\endlastfoot")
    
    # Data rows
    for row in table_rows:
        # Escape ampersands in keyword names
        kw = row['keyword'].replace('&', r'\&')
        
        line = f"{kw} & {row['pr_favor']:.1f} & {row['pr_against']:.1f} & {row['pr_neutral']:.1f} & {row['pr_n']:,} & {row['po_favor']:.1f} & {row['po_against']:.1f} & {row['po_neutral']:.1f} & {row['po_n']:,} & {row['total_n']:,} \\\\"
        latex_lines.append(line)
        latex_lines.append(r"\hline")
    
    # End table
    latex_lines.append(r"\end{longtable}")
    latex_lines.append(r"}")
    
    return "\n".join(latex_lines)

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main function."""
    print("=" * 70)
    print("GENERATING LATEX TABLE FOR STANCE DISTRIBUTION BY KEYWORD")
    print("=" * 70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_data()
    
    # Generate table data
    print("\nGenerating table data...")
    table_rows = generate_table_data(df)
    print(f"Generated data for {len(table_rows)} keywords")
    
    # Generate LaTeX
    print("\nGenerating LaTeX table...")
    latex_code = generate_latex_table(table_rows)
    
    # Save to file
    output_file = OUTPUT_DIR / "stance_distribution_table.tex"
    with open(output_file, 'w') as f:
        f.write(latex_code)
    
    print(f"\nSaved LaTeX table to: {output_file}")
    
    # Also print to console
    print("\n" + "=" * 70)
    print("LATEX TABLE OUTPUT:")
    print("=" * 70)
    print(latex_code)


if __name__ == "__main__":
    main()
