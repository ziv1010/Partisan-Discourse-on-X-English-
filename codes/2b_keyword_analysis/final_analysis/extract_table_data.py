#!/usr/bin/env python3
"""
Script to extract data for the methodology table showing:
- Aspect name
- Total tweets KeyBERT labeled for that aspect
- Pro Ruling count
- Pro Opposition count
- Percentile rank

This is for the 22 extended aspects (23-1 since one was already counted in seed).
"""

import pandas as pd
import os

# The 22 extended aspects as per keyword_ranking_analysis.py
TARGET_KEYWORDS = [
    'aatmanirbhar', 'ayodhya', 'balochistan', 'bhakts',
    'democracy', 'demonetisation', 'dictatorship', 'gdp',
    'hathras', 'inflation', 'islamists', 'lynching',
    'mahotsav', 'minorities', 'msp', 'ratetvdebate',
    'sangh', 'sharia', 'spyware', 'suicides',
    'ucc', 'unemployment'
]

def main():
    # Load the full keyword data
    input_csv = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/2b_keyword_analysis/keyword_by_stance.csv"
    df = pd.read_csv(input_csv)
    
    # Sort by Total descending and add rank
    df_sorted = df.sort_values('Total', ascending=False).reset_index(drop=True)
    df_sorted['Rank'] = df_sorted.index + 1
    total_keywords = len(df_sorted)
    
    # Calculate top 10% threshold
    top_10_pct_rank = int(total_keywords * 0.10)
    top_10_pct_min_tweets = df_sorted.iloc[top_10_pct_rank - 1]['Total']
    
    print("=" * 80)
    print("DATA EXTRACTION FOR METHODOLOGY TABLE")
    print("=" * 80)
    print(f"\nTotal keywords in dataset: {total_keywords:,}")
    print(f"Top 10% threshold: Rank <= {top_10_pct_rank:,} (>= {top_10_pct_min_tweets:,} tweets)")
    
    # Count how many keywords fall in top 10%
    top_10_count = len(df_sorted[df_sorted['Rank'] <= top_10_pct_rank])
    print(f"Keywords in top 10%: {top_10_count:,}")
    
    # Total tweets from top 10% keywords
    top_10_tweets = df_sorted[df_sorted['Rank'] <= top_10_pct_rank]['Total'].sum()
    print(f"Total tweets from top 10% keywords: {top_10_tweets:,}")
    
    # Extract data for target keywords
    results = []
    for keyword in TARGET_KEYWORDS:
        matches = df_sorted[df_sorted['keyword'].str.lower() == keyword.lower()]
        if len(matches) > 0:
            row = matches.iloc[0]
            percentile = (1 - row['Rank'] / total_keywords) * 100
            
            # Calculate actual counts from percentages
            total = row['Total']
            pro_ruling_count = int(row['Total'] * row['Pro_Ruling_Pct'] / 100)
            pro_opp_count = int(row['Total'] * row['Pro_Opposition_Pct'] / 100)
            
            results.append({
                'Aspect': row['keyword'],
                'Rank': row['Rank'],
                'Total_Tweets': row['Total'],
                'Pro_Ruling_Count': pro_ruling_count,
                'Pro_Ruling_Pct': row['Pro_Ruling_Pct'],
                'Pro_Opp_Count': pro_opp_count,
                'Pro_Opp_Pct': row['Pro_Opposition_Pct'],
                'Percentile': round(percentile, 2),
                'In_Top_10_Pct': 'Yes' if row['Rank'] <= top_10_pct_rank else 'No'
            })
    
    results_df = pd.DataFrame(results).sort_values('Rank')
    
    print("\n" + "=" * 80)
    print("TABLE DATA FOR LATEX (sorted by rank)")
    print("=" * 80)
    
    # Print header for LaTeX table
    print(f"\n{'Aspect':<18} {'Total':<8} {'Pro-Rul':<10} {'Pro-Opp':<10} {'Percentile':<12} {'Top 10%':<8}")
    print("-" * 70)
    
    for _, row in results_df.iterrows():
        print(f"{row['Aspect']:<18} {row['Total_Tweets']:<8,} {row['Pro_Ruling_Count']:<10,} {row['Pro_Opp_Count']:<10,} {row['Percentile']:<11.2f}% {row['In_Top_10_Pct']:<8}")
    
    # Summary stats
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    in_top_10 = len(results_df[results_df['In_Top_10_Pct'] == 'Yes'])
    total_target_tweets = results_df['Total_Tweets'].sum()
    print(f"\nTarget keywords in top 10%: {in_top_10} out of {len(TARGET_KEYWORDS)}")
    print(f"Total tweets across target keywords: {total_target_tweets:,}")
    
    # Check which are NOT in top 10%
    not_in_top_10 = results_df[results_df['In_Top_10_Pct'] == 'No']
    if len(not_in_top_10) > 0:
        print(f"\nKeywords NOT in top 10%: {', '.join(not_in_top_10['Aspect'].tolist())}")
    else:
        print("\nAll target keywords are in top 10%!")
    
    # Save to CSV
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "table_data_for_latex.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nData saved to: {output_path}")
    
    # Generate LaTeX table
    print("\n" + "=" * 80)
    print("LATEX TABLE CODE")
    print("=" * 80)
    print("""
\\begin{table}[h]
\\centering
\\footnotesize
\\caption{Extended Aspect Selection from Top 10\\% Keywords}
\\label{tab:extended_aspects}
\\begin{tabular}{lrrrr}
\\toprule
\\textbf{Aspect} & \\textbf{Total Tweets} & \\textbf{Pro-Ruling} & \\textbf{Pro-Opp} & \\textbf{Percentile} \\\\
\\midrule""")
    
    for _, row in results_df.iterrows():
        aspect = row['Aspect'].replace('_', '\\_')
        print(f"{aspect} & {row['Total_Tweets']:,} & {row['Pro_Ruling_Count']:,} & {row['Pro_Opp_Count']:,} & {row['Percentile']:.1f}\\% \\\\")
    
    print("""\\bottomrule
\\end{tabular}
\\end{table}""")

if __name__ == "__main__":
    main()
