#!/usr/bin/env python3
"""
Script to extract 16 unique tweets per keyword (8 pro ruling, 8 pro opposition)
from the combined stance results CSV.
"""

import pandas as pd
import os

def main():
    # Input file path
    input_csv = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/final_results+visualisations_folder/combined_stance_results.csv"
    
    # Output file path (same folder as this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_csv = os.path.join(script_dir, "extracted_tweets_per_keyword.csv")
    
    # Read the CSV
    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"Total rows: {len(df)}")
    
    # Get unique keywords
    keywords = df['keyword'].unique()
    print(f"Found {len(keywords)} unique keywords: {list(keywords)}")
    
    # Store all extracted tweets
    all_extracted = []
    
    for keyword in keywords:
        print(f"\nProcessing keyword: {keyword}")
        
        # Filter by keyword
        keyword_df = df[df['keyword'] == keyword]
        
        # Get pro ruling tweets (unique by tweet text)
        pro_ruling = keyword_df[keyword_df['_label_norm'] == 'pro ruling'].drop_duplicates(subset=['tweet'])
        pro_ruling_sample = pro_ruling.head(8)
        print(f"  Pro Ruling: Found {len(pro_ruling)} unique, taking {len(pro_ruling_sample)}")
        
        # Get pro opposition tweets (unique by tweet text)
        pro_opposition = keyword_df[keyword_df['_label_norm'] == 'pro opposition'].drop_duplicates(subset=['tweet'])
        pro_opposition_sample = pro_opposition.head(8)
        print(f"  Pro Opposition: Found {len(pro_opposition)} unique, taking {len(pro_opposition_sample)}")
        
        # Add to results
        all_extracted.append(pro_ruling_sample)
        all_extracted.append(pro_opposition_sample)
    
    # Combine all extracted tweets
    result_df = pd.concat(all_extracted, ignore_index=True)
    
    # Select relevant columns for output
    output_columns = ['keyword', 'tweet', '_label_norm', 'original_author', 'language']
    # Filter to only include columns that exist
    output_columns = [col for col in output_columns if col in result_df.columns]
    result_df = result_df[output_columns]
    
    # Save to CSV
    result_df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved {len(result_df)} tweets to {output_csv}")
    
    # Print summary
    print("\n--- Summary ---")
    for keyword in keywords:
        keyword_data = result_df[result_df['keyword'] == keyword]
        pro_ruling_count = len(keyword_data[keyword_data['_label_norm'] == 'pro ruling'])
        pro_opp_count = len(keyword_data[keyword_data['_label_norm'] == 'pro opposition'])
        print(f"{keyword}: {pro_ruling_count} pro ruling, {pro_opp_count} pro opposition")

if __name__ == "__main__":
    main()
