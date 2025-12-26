#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data EDA Script for Partisan Discourse Dataset

This script safely analyzes the large tweets_exploded_by_keyword.csv file
by reading it in chunks to avoid memory issues.

Usage:
    micromamba activate partisan_env
    python data_eda.py
"""

import pandas as pd
from pathlib import Path

# Configuration
CSV_PATH = Path("/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/final_data/tweets_exploded_by_keyword.csv")
CHUNK_SIZE = 100_000  # Read 100k rows at a time

def main():
    print("=" * 70)
    print("PARTISAN DISCOURSE DATASET - EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    
    # Get column names and sample from first chunk
    print("\n[1] Loading sample data and column structure...")
    sample_df = pd.read_csv(CSV_PATH, nrows=5)
    columns = list(sample_df.columns)
    print(f"Columns ({len(columns)}): {columns}")
    
    # Initialize counters
    total_rows = 0
    unique_keywords = set()
    unique_original_authors = set()
    unique_retweet_authors = set()
    year_counts = {}
    label_counts = {"Pro Ruling": 0, "Pro OPP": 0, "Neutral": 0, "Other": 0}
    side_counts = {"ruling": 0, "opposition": 0, "other": 0}
    
    print("\n[2] Processing dataset in chunks...")
    
    # Process in chunks
    chunk_num = 0
    for chunk in pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE, low_memory=False):
        chunk_num += 1
        total_rows += len(chunk)
        
        # Collect unique values
        if 'keyword' in chunk.columns:
            unique_keywords.update(chunk['keyword'].dropna().unique())
        if 'original_author' in chunk.columns:
            unique_original_authors.update(chunk['original_author'].dropna().unique())
        if 'retweet_author' in chunk.columns:
            unique_retweet_authors.update(chunk['retweet_author'].dropna().unique())
        
        # Year distribution
        if 'year' in chunk.columns:
            for year in chunk['year'].dropna().unique():
                year_val = str(int(float(year)))
                count = len(chunk[chunk['year'] == year])
                year_counts[year_val] = year_counts.get(year_val, 0) + count
        
        # Label distribution
        if 'tweet_label' in chunk.columns:
            for label in chunk['tweet_label'].dropna().unique():
                label_str = str(label).strip()
                if label_str in label_counts:
                    label_counts[label_str] += len(chunk[chunk['tweet_label'] == label])
                else:
                    label_counts["Other"] += len(chunk[chunk['tweet_label'] == label])
        
        # Side distribution
        if 'side' in chunk.columns:
            for side in chunk['side'].dropna().unique():
                side_str = str(side).strip().lower()
                if side_str in side_counts:
                    side_counts[side_str] += len(chunk[chunk['side'] == side])
                else:
                    side_counts["other"] += len(chunk[chunk['side'] == side])
        
        if chunk_num % 10 == 0:
            print(f"  Processed {chunk_num} chunks ({total_rows:,} rows)...")
    
    print(f"  Completed: {chunk_num} chunks ({total_rows:,} total rows)")
    
    # Print results
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    
    print(f"\n### Basic Statistics")
    print(f"  Total rows (tweet-keyword pairs): {total_rows:,}")
    print(f"  Unique keywords: {len(unique_keywords):,}")
    print(f"  Unique original authors: {len(unique_original_authors):,}")
    print(f"  Unique retweet authors: {len(unique_retweet_authors):,}")
    
    print(f"\n### Temporal Distribution (by Year)")
    for year in sorted(year_counts.keys()):
        pct = (year_counts[year] / total_rows) * 100
        print(f"  {year}: {year_counts[year]:,} ({pct:.1f}%)")
    
    print(f"\n### Tweet Label Distribution")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            pct = (count / total_rows) * 100
            print(f"  {label}: {count:,} ({pct:.1f}%)")
    
    print(f"\n### Retweeting Politician Side Distribution")
    for side, count in sorted(side_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            pct = (count / total_rows) * 100
            print(f"  {side}: {count:,} ({pct:.1f}%)")
    
    # Column descriptions for the report
    print("\n" + "=" * 70)
    print("COLUMN DESCRIPTIONS (for LaTeX table)")
    print("=" * 70)
    column_desc = {
        "timestamp": "Date and time of the tweet",
        "tweet": "Full text content of the tweet",
        "retweet_author": "Twitter handle of the politician who retweeted",
        "original_author": "Twitter handle of the original tweet author (influencer)",
        "retweet_party": "Political party of the retweeting politician",
        "year": "Year of the tweet (2020-2023)",
        "side": "Political bloc of retweeting politician (ruling/opposition)",
        "polarity_avg": "Average polarity score of the influencer (-1 to 1)",
        "tweet_label": "Inferred political leaning (Pro Ruling/Pro OPP/Neutral)",
        "keyword": "Extracted keyword/topic from the tweet",
        "subjects_scored": "KeyBERT extracted subjects with relevance scores"
    }
    
    for col in columns:
        desc = column_desc.get(col, "(description not available)")
        print(f"  {col}: {desc}")
    
    print("\n" + "=" * 70)
    print("EDA COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
