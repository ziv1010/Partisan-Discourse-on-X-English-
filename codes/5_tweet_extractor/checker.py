"""
Tweet Overlap Checker
Compares tweets across multiple extraction CSV files to identify overlaps
and unique tweets per extraction method.
"""

import pandas as pd
from pathlib import Path

# Define the CSV file paths
FILES = {
    "15kw_141k": "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/5_tweet_extractor/extracted_by_keyword_run1/extracted_ALL_15keywords_141311rows.csv",
    "23kw_48k": "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/5_tweet_extractor/exploration_before_extraction/extracted_by_keyword_run1/extracted_ALL_23keywords_48348rows.csv",
    "37kw_311k": "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/5_tweet_extractor/exploration_before_extraction/extracted_by_keyword/extracted_ALL_37keywords_311099rows.csv",
}

def load_tweets(filepath: str) -> set:
    """Load unique tweets from a CSV file."""
    df = pd.read_csv(filepath, usecols=['tweet'])
    # Use tweet text as the identifier (strip whitespace for consistency)
    tweets = set(df['tweet'].dropna().astype(str).str.strip())
    return tweets

def main():
    print("=" * 70)
    print("TWEET OVERLAP ANALYSIS")
    print("=" * 70)
    
    # Load all datasets
    datasets = {}
    for name, path in FILES.items():
        print(f"\nLoading {name}...")
        tweets = load_tweets(path)
        datasets[name] = tweets
        print(f"  → Unique tweets: {len(tweets):,}")
    
    print("\n" + "=" * 70)
    print("PAIRWISE OVERLAP ANALYSIS")
    print("=" * 70)
    
    # Pairwise comparisons
    names = list(datasets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name1, name2 = names[i], names[j]
            set1, set2 = datasets[name1], datasets[name2]
            
            overlap = set1 & set2
            only_in_1 = set1 - set2
            only_in_2 = set2 - set1
            union = set1 | set2
            
            jaccard = len(overlap) / len(union) if union else 0
            
            print(f"\n{name1} vs {name2}:")
            print(f"  Overlap (in both):     {len(overlap):>10,} tweets")
            print(f"  Only in {name1}:  {len(only_in_1):>10,} tweets")
            print(f"  Only in {name2}:  {len(only_in_2):>10,} tweets")
            print(f"  Jaccard similarity:    {jaccard:.2%}")
    
    print("\n" + "=" * 70)
    print("THREE-WAY OVERLAP ANALYSIS")
    print("=" * 70)
    
    set_15kw = datasets["15kw_141k"]
    set_23kw = datasets["23kw_48k"]
    set_37kw = datasets["37kw_311k"]
    
    # All three
    in_all_three = set_15kw & set_23kw & set_37kw
    
    # Exactly two
    only_15_and_23 = (set_15kw & set_23kw) - set_37kw
    only_15_and_37 = (set_15kw & set_37kw) - set_23kw
    only_23_and_37 = (set_23kw & set_37kw) - set_15kw
    
    # Only one
    only_15kw = set_15kw - set_23kw - set_37kw
    only_23kw = set_23kw - set_15kw - set_37kw
    only_37kw = set_37kw - set_15kw - set_23kw
    
    # Total unique
    total_unique = set_15kw | set_23kw | set_37kw
    
    print(f"\n  In ALL three files:       {len(in_all_three):>10,} tweets")
    print(f"\n  Only in 15kw & 23kw:      {len(only_15_and_23):>10,} tweets")
    print(f"  Only in 15kw & 37kw:      {len(only_15_and_37):>10,} tweets")
    print(f"  Only in 23kw & 37kw:      {len(only_23_and_37):>10,} tweets")
    print(f"\n  Unique to 15kw_141k:      {len(only_15kw):>10,} tweets")
    print(f"  Unique to 23kw_48k:       {len(only_23kw):>10,} tweets")
    print(f"  Unique to 37kw_311k:      {len(only_37kw):>10,} tweets")
    print(f"\n  TOTAL unique tweets:      {len(total_unique):>10,} tweets")
    
    print("\n" + "=" * 70)
    print("COVERAGE ANALYSIS")
    print("=" * 70)
    
    for name, tweet_set in datasets.items():
        others = set()
        for other_name, other_set in datasets.items():
            if other_name != name:
                others |= other_set
        
        unique_contribution = tweet_set - others
        covered_by_others = tweet_set & others
        
        print(f"\n  {name}:")
        print(f"    Tweets also in other files:  {len(covered_by_others):>10,} ({len(covered_by_others)/len(tweet_set)*100:.1f}%)")
        print(f"    Unique contribution:         {len(unique_contribution):>10,} ({len(unique_contribution)/len(tweet_set)*100:.1f}%)")

if __name__ == "__main__":
    main()
