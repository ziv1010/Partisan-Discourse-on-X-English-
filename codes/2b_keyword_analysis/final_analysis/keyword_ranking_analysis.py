#!/usr/bin/env python3
"""
Keyword Ranking Analysis Script

This script analyzes the ranking of specified keywords based on total tweet counts
compared to all other keywords in the dataset.

Output:
- Ranking of each target keyword (e.g., 10th out of 1000 keywords)
- Summary statistics to help identify thresholds
"""

import pandas as pd
import os

# Define the target keywords to analyze
TARGET_KEYWORDS = [
    'aatmanirbhar', 'ayodhya', 'balochistan', 'bhakts',
    'democracy', 'demonetisation', 'dictatorship', 'gdp',
    'hathras', 'inflation', 'islamists', 'lynching',
    'mahotsav', 'minorities', 'msp', 'ratetvdebate',
    'sangh', 'sharia', 'spyware', 'suicides',
    'ucc', 'unemployment'
]

def load_data(csv_path):
    """Load the keyword by stance CSV file."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} keywords from {csv_path}")
    return df

def analyze_rankings(df, target_keywords):
    """
    Analyze where target keywords rank compared to all keywords by total tweet count.
    """
    # Sort by Total in descending order
    df_sorted = df.sort_values('Total', ascending=False).reset_index(drop=True)
    
    # Add rank column (1-indexed)
    df_sorted['Rank'] = df_sorted.index + 1
    
    total_keywords = len(df_sorted)
    
    # Find rankings for target keywords
    results = []
    found_keywords = []
    not_found = []
    
    for keyword in target_keywords:
        # Search case-insensitively
        matches = df_sorted[df_sorted['keyword'].str.lower() == keyword.lower()]
        
        if len(matches) > 0:
            row = matches.iloc[0]
            results.append({
                'Keyword': row['keyword'],
                'Rank': row['Rank'],
                'Total_Tweets': row['Total'],
                'Percentile': round((1 - row['Rank'] / total_keywords) * 100, 2),
                'Pro_Ruling_Pct': row['Pro_Ruling_Pct'],
                'Pro_Opposition_Pct': row['Pro_Opposition_Pct']
            })
            found_keywords.append(keyword)
        else:
            not_found.append(keyword)
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        results_df = results_df.sort_values('Rank').reset_index(drop=True)
    
    return results_df, not_found, total_keywords, df_sorted

def generate_report(results_df, not_found, total_keywords, df_sorted, output_dir):
    """Generate analysis report with rankings and thresholds."""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("KEYWORD RANKING ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"\nTotal keywords in dataset: {total_keywords}")
    report_lines.append(f"Target keywords analyzed: {len(results_df)}")
    
    if not_found:
        report_lines.append(f"\nKeywords NOT FOUND in dataset ({len(not_found)}): {', '.join(not_found)}")
    
    report_lines.append("\n" + "-" * 80)
    report_lines.append("TARGET KEYWORD RANKINGS (sorted by rank)")
    report_lines.append("-" * 80)
    
    if len(results_df) > 0:
        report_lines.append(f"\n{'Keyword':<20} {'Rank':>8} {'Total Tweets':>15} {'Percentile':>12} {'Pro Ruling %':>14} {'Pro Opp %':>12}")
        report_lines.append("-" * 80)
        
        for _, row in results_df.iterrows():
            report_lines.append(
                f"{row['Keyword']:<20} {row['Rank']:>8} {row['Total_Tweets']:>15,} {row['Percentile']:>11.2f}% {row['Pro_Ruling_Pct']:>13.2f}% {row['Pro_Opposition_Pct']:>11.2f}%"
            )
    
    # Summary Statistics
    report_lines.append("\n" + "-" * 80)
    report_lines.append("SUMMARY STATISTICS FOR TARGET KEYWORDS")
    report_lines.append("-" * 80)
    
    if len(results_df) > 0:
        report_lines.append(f"\nRank Statistics:")
        report_lines.append(f"  - Highest ranked keyword: {results_df.iloc[0]['Keyword']} (Rank #{results_df.iloc[0]['Rank']})")
        report_lines.append(f"  - Lowest ranked keyword: {results_df.iloc[-1]['Keyword']} (Rank #{results_df.iloc[-1]['Rank']})")
        report_lines.append(f"  - Mean rank: {results_df['Rank'].mean():.1f}")
        report_lines.append(f"  - Median rank: {results_df['Rank'].median():.1f}")
        
        report_lines.append(f"\nTweet Count Statistics:")
        report_lines.append(f"  - Highest tweets: {results_df['Total_Tweets'].max():,} ({results_df.loc[results_df['Total_Tweets'].idxmax(), 'Keyword']})")
        report_lines.append(f"  - Lowest tweets: {results_df['Total_Tweets'].min():,} ({results_df.loc[results_df['Total_Tweets'].idxmin(), 'Keyword']})")
        report_lines.append(f"  - Mean tweets: {results_df['Total_Tweets'].mean():,.0f}")
        report_lines.append(f"  - Median tweets: {results_df['Total_Tweets'].median():,.0f}")
    
    # Threshold Analysis
    report_lines.append("\n" + "-" * 80)
    report_lines.append("THRESHOLD ANALYSIS")
    report_lines.append("-" * 80)
    
    # Show reference points - Top X% means the best X% (lowest rank numbers, highest tweet counts)
    percentiles = [1, 5, 10, 25, 50]
    report_lines.append(f"\nTop keyword thresholds (higher rank = lower tweet count):")
    for p in percentiles:
        threshold_rank = int(total_keywords * p / 100)
        threshold_tweets = df_sorted.iloc[threshold_rank - 1]['Total'] if threshold_rank > 0 else df_sorted.iloc[0]['Total']
        report_lines.append(f"  - Top {p}% = Rank 1 to {threshold_rank:,} (min {threshold_tweets:,} tweets)")
    
    # Distribution of target keywords by rank buckets
    report_lines.append(f"\nTarget keywords distribution by rank buckets:")
    buckets = [(1, 50), (51, 100), (101, 200), (201, 500), (501, 1000), (1001, 5000), (5001, total_keywords)]
    for low, high in buckets:
        count = len(results_df[(results_df['Rank'] >= low) & (results_df['Rank'] <= high)])
        if count > 0:
            keywords_in_bucket = results_df[(results_df['Rank'] >= low) & (results_df['Rank'] <= high)]['Keyword'].tolist()
            report_lines.append(f"  - Rank {low}-{high}: {count} keywords ({', '.join(keywords_in_bucket)})")
    
    # Reference: Top 20 overall keywords
    report_lines.append("\n" + "-" * 80)
    report_lines.append("REFERENCE: TOP 20 KEYWORDS BY TOTAL TWEETS")
    report_lines.append("-" * 80)
    
    report_lines.append(f"\n{'Rank':>6} {'Keyword':<20} {'Total Tweets':>15}")
    report_lines.append("-" * 45)
    for i in range(min(20, len(df_sorted))):
        row = df_sorted.iloc[i]
        report_lines.append(f"{row['Rank']:>6} {row['keyword']:<20} {row['Total']:>15,}")
    
    report = "\n".join(report_lines)
    
    # Save report
    report_path = os.path.join(output_dir, "keyword_ranking_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\nReport saved to: {report_path}")
    
    return report

def main():
    # Paths
    input_csv = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/2b_keyword_analysis/keyword_by_stance.csv"
    output_dir = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/2b_keyword_analysis/final_analysis"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = load_data(input_csv)
    
    # Analyze rankings
    results_df, not_found, total_keywords, df_sorted = analyze_rankings(df, TARGET_KEYWORDS)
    
    # Generate and save report
    generate_report(results_df, not_found, total_keywords, df_sorted, output_dir)
    
    # Save detailed results to CSV
    if len(results_df) > 0:
        results_csv_path = os.path.join(output_dir, "keyword_ranking_results.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"\nResults saved to: {results_csv_path}")

if __name__ == "__main__":
    main()
