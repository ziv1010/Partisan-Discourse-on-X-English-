#!/usr/bin/env python3
"""
Keyword Count Analysis Script

This script analyzes keyword coverage in tweets by comparing:
1. Keyword column search: Tweets that have the keyword in the 'keyword' column
2. Fallback search: Additional tweets that contain the keyword in the tweet text 
   (but are NOT already found via keyword column)
3. Total: Combined unique tweets from both methods

Since tweets can span multiple CSV rows (due to embedded newlines), we identify 
unique tweets by a composite key of (timestamp, retweet_author, original_author).
"""

import pandas as pd
import re
from pathlib import Path
from typing import List, Set, Tuple
import warnings
warnings.filterwarnings('ignore')

# Define the keyword list
KEYWORDS = [
    "aatmanirbhar",
    "ayodhya",
    "balochistan",
    "bhakts",
    "caa",
    "china",
    "congress",
    "democracy",
    "demonetisation",
    "dictatorship",
    "farm laws",
    "farmers protests",
    "gdp",
    "hathras",
    "hindu",
    "hindutva",
    "inflation",
    "islamists",
    "kashmir",
    "kashmiri pandits",
    "lynching",
    "mahotsav",
    "minorities",
    "modi",
    "msp",
    "muslim",
    "new parliament",
    "rahulgandhi",
    "ram mandir",
    "ratetvdebate",
    "sangh",
    "shaheen bagh",
    "sharia",
    "spyware",
    "suicides",
    "ucc",
    "unemployment",
]


def create_tweet_id(row: pd.Series) -> str:
    """Create a unique identifier for a tweet using timestamp and authors."""
    return f"{row['timestamp']}_{row['retweet_author']}_{row['original_author']}"


def load_data(filepath: str) -> pd.DataFrame:
    """Load the CSV file and add tweet identifiers."""
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Total rows loaded: {len(df):,}")
    
    # Create unique tweet identifier
    df['tweet_id'] = df.apply(create_tweet_id, axis=1)
    
    # Get unique tweets count
    unique_tweets = df['tweet_id'].nunique()
    print(f"Unique tweets identified: {unique_tweets:,}")
    
    return df


def get_keyword_column_tweets(df: pd.DataFrame, keyword: str) -> Set[str]:
    """Get unique tweet IDs where keyword appears in the 'keyword' column."""
    mask = df['keyword'].str.lower() == keyword.lower()
    return set(df.loc[mask, 'tweet_id'].unique())


def get_fallback_text_tweets(df: pd.DataFrame, keyword: str, exclude_ids: Set[str]) -> Set[str]:
    """
    Get unique tweet IDs where keyword appears in tweet text,
    EXCLUDING tweets already found via keyword column search.
    
    Uses case-insensitive search with word boundary matching.
    """
    # Create regex pattern for word boundary matching (case insensitive)
    # Escape special regex characters in keyword
    escaped_keyword = re.escape(keyword)
    pattern = rf'\b{escaped_keyword}\b'
    
    # Search in tweet text (case insensitive)
    # Handle NaN values
    mask = df['tweet'].fillna('').str.contains(pattern, case=False, regex=True, na=False)
    
    # Get all matching tweet IDs
    matching_ids = set(df.loc[mask, 'tweet_id'].unique())
    
    # Exclude tweets already found via keyword column
    fallback_only_ids = matching_ids - exclude_ids
    
    return fallback_only_ids


def analyze_keywords(df: pd.DataFrame, keywords: List[str]) -> pd.DataFrame:
    """
    Analyze each keyword and return counts for:
    - keyword_column_count: Tweets found via keyword column
    - fallback_count: Additional tweets found via text search (non-overlapping)
    - total_count: Combined unique tweets
    """
    results = []
    
    print("\nAnalyzing keywords...")
    print("-" * 80)
    
    for i, keyword in enumerate(keywords, 1):
        print(f"[{i}/{len(keywords)}] Processing: {keyword}")
        
        # Get tweets from keyword column
        keyword_tweets = get_keyword_column_tweets(df, keyword)
        keyword_count = len(keyword_tweets)
        
        # Get additional tweets from text search (excluding keyword column matches)
        fallback_tweets = get_fallback_text_tweets(df, keyword, keyword_tweets)
        fallback_count = len(fallback_tweets)
        
        # Total is the union of both sets (already non-overlapping)
        total_count = keyword_count + fallback_count
        
        results.append({
            'keyword': keyword,
            'keyword_column_count': keyword_count,
            'fallback_text_count': fallback_count,
            'total_count': total_count
        })
        
        print(f"    Keyword column: {keyword_count:,} | Fallback: {fallback_count:,} | Total: {total_count:,}")
    
    print("-" * 80)
    
    return pd.DataFrame(results)


def main():
    """Main function to run the keyword count analysis."""
    # File paths
    input_file = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/final_data/tweets_exploded_by_keyword.csv"
    output_dir = Path("/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/2b_keyword_analysis")
    output_file = output_dir / "keyword_count_results.csv"
    
    # Load data
    df = load_data(input_file)
    
    # Analyze keywords
    results_df = analyze_keywords(df, KEYWORDS)
    
    # Calculate totals
    total_keyword_column = results_df['keyword_column_count'].sum()
    total_fallback = results_df['fallback_text_count'].sum()
    total_combined = results_df['total_count'].sum()
    
    # Add totals row
    totals_row = pd.DataFrame([{
        'keyword': 'TOTAL',
        'keyword_column_count': total_keyword_column,
        'fallback_text_count': total_fallback,
        'total_count': total_combined
    }])
    
    results_df = pd.concat([results_df, totals_row], ignore_index=True)
    
    # Save results
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Keyword':<25} {'Keyword Column':>18} {'Fallback Text':>18} {'Total':>12}")
    print("-" * 80)
    
    for _, row in results_df.iterrows():
        print(f"{row['keyword']:<25} {row['keyword_column_count']:>18,} {row['fallback_text_count']:>18,} {row['total_count']:>12,}")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
