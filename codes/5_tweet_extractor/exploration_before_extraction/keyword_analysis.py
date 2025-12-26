"""
Keyword Analysis Script
Analyzes tweet counts for specified keywords using:
1) keyword column matching (case-insensitive)
2) fallback tweet text search (case-insensitive)
Also provides pro-ruling vs pro-opposition split.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import time

# Define keywords grouped by category
KEYWORDS = {
    "Religious/Nationalist": [
        "hindus", "hindutva", "rss", "sangh", "rammandir", "ayodhya", "jaishriram",
        "ucc", "tripletalaq", "balakotstrike", "surgicalstrike", "kashmir",
        "islamists", "sharia", "urduwood", "balochistan", "sarkaryavah"
    ],
    "Government Schemes/Pro-Modi": [
        "aatmanirbhar", "vocal4local", "swachhbharat", "digitalindia", "makeinindia",
        "sabkasaath", "sabkavikas", "ujjwala", "jandhan", "ayushmanbharat", "pmay",
        "mudra", "gst", "caa", "nrc", "article370", "vandebharat", "g20india",
        "harghartiranga", "amritkaal", "newindia", "vaccination", "mahotsav",
        "millets", "nep2020"
    ],
    "Economic/Social Issues": [
        "unemployment", "berozgari", "jobscrisis", "inflation", "mehengai", "pricerise",
        "gascylinder", "petrol", "diesel", "suicides", "farmersprotest", "kisanandolan",
        "msp", "gdp", "womensafety", "hathras", "dalitrights", "minorities", "muslims"
    ],
    "Criticism/Scandals": [
        "adani", "ambani", "hindenburg", "cronycapitalism", "manipurviolence", "lynching",
        "hatecrime", "godimedia", "pegasus", "spyware", "snooping", "pmcares",
        "oxygencrisis", "demonetisation", "notebandi", "lakhimpurkheri", "catastrophe",
        "electoralbonds", "edraid", "agencymisuse"
    ],
    "Opposition/Criticism": [
        "saveconstitution", "democracy", "authoritarian", "dictatorship", "lapdog",
        "bhakts", "teleprompter", "trolls", "supremacists", "darpok56inch", "shaheenbagh",
        "caaprotest", "ratetvdebate", "thewire_in"
    ]
}

# Flatten keywords list
ALL_KEYWORDS = []
for category, kws in KEYWORDS.items():
    ALL_KEYWORDS.extend(kws)


def analyze_keywords(data_path: str, output_path: str):
    """Analyze keywords in the tweet dataset."""
    
    print(f"Loading data from {data_path}...")
    start_time = time.time()
    
    # Load only necessary columns for efficiency
    df = pd.read_csv(
        data_path,
        usecols=['tweet', 'keyword', 'tweet_label'],
        dtype={'keyword': 'str', 'tweet_label': 'str', 'tweet': 'str'}
    )
    
    print(f"Data loaded in {time.time() - start_time:.1f}s. Shape: {df.shape}")
    
    # Lowercase keyword column for case-insensitive matching
    df['keyword_lower'] = df['keyword'].str.lower()
    
    # Prepare results storage
    results = []
    
    print(f"\nAnalyzing {len(ALL_KEYWORDS)} keywords...")
    
    # Get unique tweets for fallback search (to avoid counting same tweet multiple times)
    # We need to work with unique tweet-label pairs
    unique_tweets = df.drop_duplicates(subset=['tweet'])[['tweet', 'tweet_label']].copy()
    unique_tweets['tweet_lower'] = unique_tweets['tweet'].str.lower().fillna('')
    print(f"Unique tweets for fallback search: {len(unique_tweets)}")
    
    for i, keyword in enumerate(ALL_KEYWORDS):
        kw_lower = keyword.lower()
        
        # Method 1: Keyword column match (case-insensitive)
        keyword_mask = df['keyword_lower'] == kw_lower
        keyword_count = keyword_mask.sum()
        
        # Split by stance for keyword column
        keyword_df = df[keyword_mask]
        keyword_pro_ruling = (keyword_df['tweet_label'] == 'Pro Ruling').sum()
        keyword_pro_opp = (keyword_df['tweet_label'] == 'Pro OPP').sum()
        keyword_neutral = (keyword_df['tweet_label'] == 'Neutral').sum()
        
        # Method 2: Fallback - search in tweet text (case-insensitive)
        # Count unique tweets containing the keyword
        tweet_mask = unique_tweets['tweet_lower'].str.contains(kw_lower, case=False, na=False, regex=False)
        fallback_count = tweet_mask.sum()
        
        # Split by stance for fallback
        fallback_df = unique_tweets[tweet_mask]
        fallback_pro_ruling = (fallback_df['tweet_label'] == 'Pro Ruling').sum()
        fallback_pro_opp = (fallback_df['tweet_label'] == 'Pro OPP').sum()
        fallback_neutral = (fallback_df['tweet_label'] == 'Neutral').sum()
        
        # Find category
        category = None
        for cat, kws in KEYWORDS.items():
            if keyword in kws:
                category = cat
                break
        
        results.append({
            'keyword': keyword,
            'category': category,
            # Keyword column stats
            'keyword_col_count': keyword_count,
            'keyword_col_pro_ruling': keyword_pro_ruling,
            'keyword_col_pro_opp': keyword_pro_opp,
            'keyword_col_neutral': keyword_neutral,
            # Fallback (tweet text) stats
            'fallback_count': fallback_count,
            'fallback_pro_ruling': fallback_pro_ruling,
            'fallback_pro_opp': fallback_pro_opp,
            'fallback_neutral': fallback_neutral,
        })
        
        if (i + 1) % 10 == 0 or (i + 1) == len(ALL_KEYWORDS):
            print(f"  Processed {i + 1}/{len(ALL_KEYWORDS)} keywords...")
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Add percentage columns
    for prefix in ['keyword_col', 'fallback']:
        total_col = f'{prefix}_count'
        for stance in ['pro_ruling', 'pro_opp', 'neutral']:
            count_col = f'{prefix}_{stance}'
            pct_col = f'{prefix}_{stance}_pct'
            results_df[pct_col] = (results_df[count_col] / results_df[total_col] * 100).round(2).fillna(0)
    
    # Reorder columns
    column_order = [
        'keyword', 'category',
        'keyword_col_count', 'keyword_col_pro_ruling', 'keyword_col_pro_ruling_pct',
        'keyword_col_pro_opp', 'keyword_col_pro_opp_pct', 'keyword_col_neutral', 'keyword_col_neutral_pct',
        'fallback_count', 'fallback_pro_ruling', 'fallback_pro_ruling_pct',
        'fallback_pro_opp', 'fallback_pro_opp_pct', 'fallback_neutral', 'fallback_neutral_pct'
    ]
    results_df = results_df[column_order]
    
    # Sort by keyword column count descending
    results_df = results_df.sort_values('keyword_col_count', ascending=False)
    
    # Save results
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    # Print summary
    print("\n" + "="*100)
    print("KEYWORD ANALYSIS SUMMARY")
    print("="*100)
    
    # Print by category
    for category in KEYWORDS.keys():
        cat_df = results_df[results_df['category'] == category].copy()
        print(f"\n{'='*80}")
        print(f"Category: {category}")
        print(f"{'='*80}")
        print(f"{'Keyword':<20} {'KW Col':>10} {'KW Ruling':>12} {'KW Opp':>10} {'Fallback':>12} {'FB Ruling':>12} {'FB Opp':>10}")
        print("-"*80)
        
        for _, row in cat_df.iterrows():
            print(f"{row['keyword']:<20} {row['keyword_col_count']:>10,} "
                  f"{row['keyword_col_pro_ruling']:>8,}({row['keyword_col_pro_ruling_pct']:>4.1f}%) "
                  f"{row['keyword_col_pro_opp']:>6,}({row['keyword_col_pro_opp_pct']:>4.1f}%) "
                  f"{row['fallback_count']:>12,} "
                  f"{row['fallback_pro_ruling']:>8,}({row['fallback_pro_ruling_pct']:>4.1f}%) "
                  f"{row['fallback_pro_opp']:>6,}({row['fallback_pro_opp_pct']:>4.1f}%)")
    
    # Overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    print(f"Total keywords analyzed: {len(ALL_KEYWORDS)}")
    print(f"Keywords with > 0 matches in keyword column: {(results_df['keyword_col_count'] > 0).sum()}")
    print(f"Keywords with > 0 matches in fallback search: {(results_df['fallback_count'] > 0).sum()}")
    print(f"Total keyword column matches: {results_df['keyword_col_count'].sum():,}")
    print(f"Total fallback matches (unique tweets): {results_df['fallback_count'].sum():,}")
    
    print(f"\nTotal time: {time.time() - start_time:.1f}s")
    
    return results_df


if __name__ == "__main__":
    DATA_PATH = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/final_data/tweets_exploded_by_keyword.csv"
    OUTPUT_PATH = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/5_tweet_extractor/exploration_before_extraction/keyword_analysis_results.csv"
    
    results = analyze_keywords(DATA_PATH, OUTPUT_PATH)
