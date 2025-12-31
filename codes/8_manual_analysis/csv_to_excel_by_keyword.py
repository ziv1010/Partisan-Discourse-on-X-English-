"""
Script to convert stance results CSV to Excel with separate sheets per keyword.
"""

import pandas as pd
from pathlib import Path

# File paths
INPUT_CSV = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/6_stance/results/final_en_results/stance_results_37keywords.csv"
OUTPUT_EXCEL = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/8_manual_analysis/stance_results_by_keyword.xlsx"


def main():
    print(f"Reading CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    
    # Get unique keywords
    keywords = df['keyword'].unique()
    print(f"Found {len(keywords)} unique keywords")
    
    # Create Excel writer
    with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
        for keyword in sorted(keywords):
            # Filter data for this keyword
            keyword_df = df[df['keyword'] == keyword].copy()
            
            # Excel sheet names have max 31 chars, clean special characters
            sheet_name = str(keyword)[:31].replace('/', '_').replace('\\', '_').replace('*', '_')
            
            # Write to sheet
            keyword_df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  Written sheet '{sheet_name}' with {len(keyword_df)} rows")
    
    print(f"\nExcel file saved to: {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()
