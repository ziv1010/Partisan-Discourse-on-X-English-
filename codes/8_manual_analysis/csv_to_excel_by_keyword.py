"""
Script to convert stance results CSV to multiple Excel files with separate sheets per keyword.
Creates multiple Excel files with a maximum of 10 keywords each to ensure all keywords are captured.
"""

import pandas as pd
from pathlib import Path
import math

# File paths
INPUT_CSV = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/6_stance/results/final_en_results/stance_results_37keywords.csv"
OUTPUT_DIR = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/8_manual_analysis"
OUTPUT_BASE_NAME = "stance_results_by_keyword"

# Maximum keywords per Excel file
MAX_KEYWORDS_PER_FILE = 10


def main():
    print(f"Reading CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    
    # Get unique keywords and sort them
    keywords = sorted(df['keyword'].unique())
    total_keywords = len(keywords)
    print(f"Found {total_keywords} unique keywords")
    
    # Calculate number of files needed
    num_files = math.ceil(total_keywords / MAX_KEYWORDS_PER_FILE)
    print(f"Will create {num_files} Excel file(s) with max {MAX_KEYWORDS_PER_FILE} keywords each\n")
    
    # Track total keywords processed
    total_keywords_processed = 0
    all_keyword_list = []
    
    # Split keywords into chunks and create separate Excel files
    for file_idx in range(num_files):
        start_idx = file_idx * MAX_KEYWORDS_PER_FILE
        end_idx = min((file_idx + 1) * MAX_KEYWORDS_PER_FILE, total_keywords)
        keywords_chunk = keywords[start_idx:end_idx]
        
        # Create filename with part number
        if num_files == 1:
            output_file = f"{OUTPUT_DIR}/{OUTPUT_BASE_NAME}.xlsx"
        else:
            output_file = f"{OUTPUT_DIR}/{OUTPUT_BASE_NAME}_part{file_idx + 1}.xlsx"
        
        print(f"Creating file {file_idx + 1}/{num_files}: {Path(output_file).name}")
        print(f"  Keywords {start_idx + 1} to {end_idx} of {total_keywords}")
        
        # Create Excel writer for this chunk
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            for keyword in keywords_chunk:
                # Filter data for this keyword
                keyword_df = df[df['keyword'] == keyword].copy()
                
                # Excel sheet names have max 31 chars, clean special characters
                sheet_name = str(keyword)[:31].replace('/', '_').replace('\\', '_').replace('*', '_')
                
                # Write to sheet
                keyword_df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"    Written sheet '{sheet_name}' with {len(keyword_df)} rows")
                
                total_keywords_processed += 1
                all_keyword_list.append(keyword)
        
        print(f"  Saved: {output_file}\n")
    
    # Final summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total keywords in CSV: {total_keywords}")
    print(f"Total keywords processed: {total_keywords_processed}")
    print(f"Files created: {num_files}")
    print(f"\nAll keywords processed:")
    for i, kw in enumerate(all_keyword_list, 1):
        print(f"  {i}. {kw}")
    
    if total_keywords_processed == total_keywords:
        print(f"\n✓ SUCCESS: All {total_keywords} keywords have been written to Excel files!")
    else:
        print(f"\n✗ WARNING: Mismatch! Expected {total_keywords}, processed {total_keywords_processed}")


if __name__ == "__main__":
    main()
