#!/usr/bin/env python3
"""
Proportion-Based Significance Testing for Stance Analysis

Uses a z-test for proportions to compare the PERCENTAGE of each stance
between groups. This accounts for imbalanced sample sizes.

For each keyword, compares:
- % Favor in pro-ruling vs % Favor in pro-opposition
- % Against in pro-ruling vs % Against in pro-opposition  
- % Neutral in pro-ruling vs % Neutral in pro-opposition
"""

import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
import warnings
import os

warnings.filterwarnings('ignore')

# File paths
ENGLISH_DATA = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/final_results+visualisations_folder/stance_results_37keywords.csv"
HINDI_DATA = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/final_results+visualisations_folder/hindi_stance_results.csv"
OUTPUT_DIR = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/last minute work/significance_testing/proportion_based"


def load_data():
    """Load English and Hindi datasets."""
    print("Loading datasets...")
    
    df_english = pd.read_csv(ENGLISH_DATA)
    df_english['language'] = 'english'
    print(f"  English dataset: {len(df_english):,} rows")
    
    df_hindi = pd.read_csv(HINDI_DATA)
    df_hindi['language'] = 'hindi'
    print(f"  Hindi dataset: {len(df_hindi):,} rows")
    
    return df_english, df_hindi


def get_stance_stats(df, keyword, political_leaning):
    """
    Get stance counts and percentages for a keyword and political leaning.
    """
    mask = (df['keyword'] == keyword) & (df['_label_norm'] == political_leaning)
    subset = df[mask]
    total = len(subset)
    
    if total == 0:
        return None
    
    stance_col = 'fewshot_label_for_against'
    stats = {
        'total': total,
        'favor_count': len(subset[subset[stance_col] == 'favor']),
        'against_count': len(subset[subset[stance_col] == 'against']),
        'neutral_count': len(subset[subset[stance_col] == 'neutral']),
    }
    stats['favor_pct'] = stats['favor_count'] / total * 100
    stats['against_pct'] = stats['against_count'] / total * 100
    stats['neutral_pct'] = stats['neutral_count'] / total * 100
    
    return stats


def proportion_ztest(count1, total1, count2, total2):
    """
    Perform z-test for proportions.
    
    Tests if the proportion count1/total1 differs from count2/total2.
    Returns: (z_statistic, p_value, significant)
    """
    try:
        if total1 < 10 or total2 < 10:
            return np.nan, np.nan, False
        
        counts = np.array([count1, count2])
        totals = np.array([total1, total2])
        
        z_stat, p_value = proportions_ztest(counts, totals, alternative='two-sided')
        return z_stat, p_value, p_value < 0.05
    except Exception:
        return np.nan, np.nan, False


def run_intra_dataset_tests(df, language):
    """
    Run proportion z-tests for each keyword.
    Compares stance percentages between pro-ruling vs pro-opposition.
    """
    print(f"\nRunning proportion-based tests for {language}...")
    
    keywords = df['keyword'].unique()
    results = []
    
    for keyword in keywords:
        stats_ruling = get_stance_stats(df, keyword, 'pro ruling')
        stats_opp = get_stance_stats(df, keyword, 'pro opposition')
        
        if stats_ruling is None or stats_opp is None:
            continue
        
        if stats_ruling['total'] < 10 or stats_opp['total'] < 10:
            continue
        
        for stance in ['favor', 'against', 'neutral']:
            z_stat, p_value, significant = proportion_ztest(
                stats_ruling[f'{stance}_count'], stats_ruling['total'],
                stats_opp[f'{stance}_count'], stats_opp['total']
            )
            
            results.append({
                'keyword': keyword,
                'stance': stance,
                'test_method': 'proportion_ztest',
                'z_statistic': z_stat,
                'p_value': p_value,
                'significant': significant,
                'pro_ruling_pct': round(stats_ruling[f'{stance}_pct'], 2),
                'pro_opp_pct': round(stats_opp[f'{stance}_pct'], 2),
                'pct_difference': round(stats_ruling[f'{stance}_pct'] - stats_opp[f'{stance}_pct'], 2),
                'n_pro_ruling': stats_ruling['total'],
                'n_pro_opposition': stats_opp['total']
            })
    
    print(f"  Processed {len(keywords)} keywords, generated {len(results)} test results")
    return pd.DataFrame(results)


def run_inter_dataset_tests(df_english, df_hindi):
    """
    Run proportion z-tests comparing English vs Hindi percentages.
    """
    print("\nRunning inter-dataset proportion tests...")
    
    results = []
    
    english_keywords = set(df_english['keyword'].unique())
    hindi_keywords = set(df_hindi['keyword'].unique())
    common_keywords = english_keywords.intersection(hindi_keywords)
    
    print(f"  Found {len(common_keywords)} common keywords")
    
    for political_leaning in ['pro ruling', 'pro opposition']:
        for keyword in common_keywords:
            stats_en = get_stance_stats(df_english, keyword, political_leaning)
            stats_hi = get_stance_stats(df_hindi, keyword, political_leaning)
            
            if stats_en is None or stats_hi is None:
                continue
            
            if stats_en['total'] < 10 or stats_hi['total'] < 10:
                continue
            
            for stance in ['favor', 'against', 'neutral']:
                z_stat, p_value, significant = proportion_ztest(
                    stats_en[f'{stance}_count'], stats_en['total'],
                    stats_hi[f'{stance}_count'], stats_hi['total']
                )
                
                results.append({
                    'keyword': keyword,
                    'political_leaning': political_leaning,
                    'stance': stance,
                    'test_method': 'proportion_ztest',
                    'z_statistic': z_stat,
                    'p_value': p_value,
                    'significant': significant,
                    'english_pct': round(stats_en[f'{stance}_pct'], 2),
                    'hindi_pct': round(stats_hi[f'{stance}_pct'], 2),
                    'pct_difference': round(stats_en[f'{stance}_pct'] - stats_hi[f'{stance}_pct'], 2),
                    'n_english': stats_en['total'],
                    'n_hindi': stats_hi['total']
                })
    
    print(f"  Generated {len(results)} inter-dataset test results")
    return pd.DataFrame(results)


def generate_summary_report(intra_english_df, intra_hindi_df, inter_df):
    """Generate summary report with percentage differences."""
    
    lines = []
    lines.append("=" * 80)
    lines.append("PROPORTION-BASED SIGNIFICANCE TESTING SUMMARY")
    lines.append("(Comparing stance PERCENTAGES between groups)")
    lines.append("=" * 80)
    lines.append("")
    
    lines.append("OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"English: {len(intra_english_df)} tests, {intra_english_df['significant'].sum()} significant ({intra_english_df['significant'].mean()*100:.1f}%)")
    lines.append(f"Hindi: {len(intra_hindi_df)} tests, {intra_hindi_df['significant'].sum()} significant ({intra_hindi_df['significant'].mean()*100:.1f}%)")
    lines.append(f"Inter-dataset: {len(inter_df)} tests, {inter_df['significant'].sum()} significant ({inter_df['significant'].mean()*100:.1f}%)")
    lines.append("")
    
    # English by stance with percentage differences
    lines.append("=" * 80)
    lines.append("ENGLISH: Largest Percentage Differences (Significant only)")
    lines.append("-" * 40)
    
    for stance in ['favor', 'against', 'neutral']:
        sig = intra_english_df[(intra_english_df['significant'] == True) & (intra_english_df['stance'] == stance)]
        sig = sig.sort_values('pct_difference', key=abs, ascending=False)
        lines.append(f"\n  {stance.upper()} stance:")
        for _, row in sig.head(10).iterrows():
            diff_sign = "+" if row['pct_difference'] > 0 else ""
            lines.append(f"    {row['keyword']}: {row['pro_ruling_pct']:.1f}% vs {row['pro_opp_pct']:.1f}% ({diff_sign}{row['pct_difference']:.1f}pp), p={row['p_value']:.2e}")
    
    # Hindi
    lines.append("")
    lines.append("=" * 80)
    lines.append("HINDI: Largest Percentage Differences (Significant only)")
    lines.append("-" * 40)
    
    for stance in ['favor', 'against', 'neutral']:
        sig = intra_hindi_df[(intra_hindi_df['significant'] == True) & (intra_hindi_df['stance'] == stance)]
        sig = sig.sort_values('pct_difference', key=abs, ascending=False)
        lines.append(f"\n  {stance.upper()} stance:")
        for _, row in sig.head(10).iterrows():
            diff_sign = "+" if row['pct_difference'] > 0 else ""
            lines.append(f"    {row['keyword']}: {row['pro_ruling_pct']:.1f}% vs {row['pro_opp_pct']:.1f}% ({diff_sign}{row['pct_difference']:.1f}pp), p={row['p_value']:.2e}")
    
    # Methodology
    lines.append("")
    lines.append("=" * 80)
    lines.append("METHODOLOGY")
    lines.append("-" * 40)
    lines.append("- Test: Z-test for proportions (two-sided)")
    lines.append("- Compares the PERCENTAGE of tweets with each stance")
    lines.append("- Accounts for different sample sizes (imbalanced data)")
    lines.append("- Significance threshold: p < 0.05")
    lines.append("- 'pp' = percentage points difference")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def main():
    """Main execution."""
    print("=" * 60)
    print("PROPORTION-BASED SIGNIFICANCE TESTING")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df_english, df_hindi = load_data()
    
    # Run tests
    intra_english_df = run_intra_dataset_tests(df_english, 'English')
    intra_hindi_df = run_intra_dataset_tests(df_hindi, 'Hindi')
    inter_df = run_inter_dataset_tests(df_english, df_hindi)
    
    # Save results
    print("\nSaving results...")
    
    intra_english_df.to_csv(os.path.join(OUTPUT_DIR, "intra_english_proportions.csv"), index=False)
    print(f"  Saved: intra_english_proportions.csv")
    
    intra_hindi_df.to_csv(os.path.join(OUTPUT_DIR, "intra_hindi_proportions.csv"), index=False)
    print(f"  Saved: intra_hindi_proportions.csv")
    
    inter_df.to_csv(os.path.join(OUTPUT_DIR, "inter_dataset_proportions.csv"), index=False)
    print(f"  Saved: inter_dataset_proportions.csv")
    
    summary = generate_summary_report(intra_english_df, intra_hindi_df, inter_df)
    with open(os.path.join(OUTPUT_DIR, "summary_report.txt"), 'w') as f:
        f.write(summary)
    print(f"  Saved: summary_report.txt")
    
    print("\n" + summary)
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
