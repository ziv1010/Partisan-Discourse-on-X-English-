#!/usr/bin/env python3
"""
Merge stance_results_37keywords.csv and hindi_stance_results.csv into a single CSV.
Handles column differences by keeping common columns and adding a 'language' identifier.
"""

import pandas as pd
from pathlib import Path

def main():
    base_path = Path(__file__).parent
    
    # Read both CSVs
    english_df = pd.read_csv(base_path / "stance_results_37keywords.csv")
    hindi_df = pd.read_csv(base_path / "hindi_stance_results_clean_consistent.csv")
    
    print(f"English CSV shape: {english_df.shape}")
    print(f"English columns: {list(english_df.columns)}")
    print(f"\nHindi CSV shape: {hindi_df.shape}")
    print(f"Hindi columns: {list(hindi_df.columns)}")
    
    # Add language identifier
    english_df['language'] = 'english'
    hindi_df['language'] = 'hindi'
    
    # Drop the duplicate source_row.1 column from hindi if it exists
    if 'source_row.1' in hindi_df.columns:
        hindi_df = hindi_df.drop(columns=['source_row.1'])
    
    # Find common columns
    common_cols = list(set(english_df.columns) & set(hindi_df.columns))
    print(f"\nCommon columns: {common_cols}")
    
    # Columns only in English
    english_only = list(set(english_df.columns) - set(hindi_df.columns))
    print(f"Columns only in English: {english_only}")
    
    # Columns only in Hindi
    hindi_only = list(set(hindi_df.columns) - set(english_df.columns))
    print(f"Columns only in Hindi: {hindi_only}")
    
    # Add missing columns with NaN
    for col in english_only:
        hindi_df[col] = pd.NA
    for col in hindi_only:
        english_df[col] = pd.NA
    
    # Ensure same column order
    all_cols = list(english_df.columns)
    hindi_df = hindi_df[all_cols]
    
    # Concatenate
    combined_df = pd.concat([english_df, hindi_df], ignore_index=True)
    
    print(f"\nCombined CSV shape: {combined_df.shape}")
    print(f"Combined columns: {list(combined_df.columns)}")
    
    # Language distribution
    print(f"\nLanguage distribution:")
    print(combined_df['language'].value_counts())
    
    # Save combined CSV
    output_path = base_path / "combined_stance_results.csv"
    combined_df.to_csv(output_path, index=False)
    print(f"\nSaved combined CSV to: {output_path}")

if __name__ == "__main__":
    main()
