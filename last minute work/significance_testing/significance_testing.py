#!/usr/bin/env python3
"""
Significance Testing for Stance Analysis (Mixed Effects Logistic Regression)

Uses binomial GLM to test if influencer alignment significantly predicts
the count of "favor" tweets for each aspect (keyword).

Model: logit(p_i) = β0 + β1 * Alignment_i + u_i
- p_i: probability of favor for influencer i
- Alignment_i: 0 for pro-ruling, 1 for opposition
- i: influencer (original_author)

Analysis is done separately per aspect (keyword).
If β1 ≠ 0 and p < 0.05, influencer alignment significantly affects favor tweet counts.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial
import warnings
import os

warnings.filterwarnings('ignore')

# File paths
COMBINED_DATA = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/final_results+visualisations_folder/combined_stance_results.csv"
OUTPUT_DIR = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/last minute work/significance_testing"


def load_data():
    """Load the combined dataset."""
    print("Loading dataset...")
    df = pd.read_csv(COMBINED_DATA)
    print(f"  Total rows: {len(df):,}")
    print(f"  Unique keywords: {df['keyword'].nunique()}")
    print(f"  Unique influencers: {df['original_author'].nunique()}")
    return df


def aggregate_by_influencer(df, keyword):
    """
    For a given keyword, aggregate data by influencer.
    
    Returns DataFrame with:
    - original_author: influencer identifier
    - alignment: 0 for pro-ruling, 1 for opposition
    - n_favor: count of favor tweets
    - n_total: total tweet count
    """
    # Filter for this keyword
    subset = df[df['keyword'] == keyword].copy()
    
    if len(subset) == 0:
        return None
    
    # Group by influencer
    grouped = subset.groupby('original_author').agg({
        '_label_norm': 'first',  # Assume consistent alignment per influencer
        'fewshot_label_for_against': lambda x: (x == 'favor').sum(),  # n_favor
        'keyword': 'count'  # n_total
    }).reset_index()
    
    grouped.columns = ['original_author', 'alignment_raw', 'n_favor', 'n_total']
    
    # Encode alignment: 0 for pro-ruling, 1 for opposition
    grouped['alignment'] = (grouped['alignment_raw'] == 'pro opposition').astype(int)
    
    # Calculate n_not_favor for binomial response
    grouped['n_not_favor'] = grouped['n_total'] - grouped['n_favor']
    
    return grouped


def fit_binomial_glm(influencer_data):
    """
    Fit binomial GLM: logit(p_i) = β0 + β1 * Alignment_i
    
    Returns: (beta1, std_err, z_value, p_value, n_influencers, n_ruling, n_opp)
    """
    try:
        # Filter out influencers with zero tweets
        data = influencer_data[influencer_data['n_total'] > 0].copy()
        
        if len(data) < 5:
            return None
        
        # Count influencers per alignment
        n_ruling = (data['alignment'] == 0).sum()
        n_opp = (data['alignment'] == 1).sum()
        
        if n_ruling < 2 or n_opp < 2:
            return None
        
        # Check for variation in outcome
        if data['n_favor'].sum() == 0 or data['n_favor'].sum() == data['n_total'].sum():
            return None
        
        # Prepare response variable (success, failure counts)
        endog = data[['n_favor', 'n_not_favor']].values
        
        # Prepare predictor with constant
        exog = sm.add_constant(data['alignment'])
        
        # Fit binomial GLM
        model = sm.GLM(endog, exog, family=Binomial())
        result = model.fit(disp=0)
        
        # Extract β1 coefficient (for alignment)
        beta1 = result.params.iloc[1]  # Alignment coefficient
        std_err = result.bse.iloc[1]
        z_value = result.tvalues.iloc[1]
        p_value = result.pvalues.iloc[1]
        
        return {
            'beta1': beta1,
            'std_err': std_err,
            'z_value': z_value,
            'p_value': p_value,
            'n_influencers': len(data),
            'n_pro_ruling_influencers': n_ruling,
            'n_opposition_influencers': n_opp,
            'total_tweets': data['n_total'].sum(),
            'total_favor': data['n_favor'].sum()
        }
        
    except Exception as e:
        print(f"    Error fitting model: {e}")
        return None


def interpret_result(row):
    """Interpret the β1 coefficient."""
    if pd.isna(row['p_value']) or not row['significant']:
        return "No significant difference"
    
    if row['beta1'] > 0:
        # Positive β1 means opposition (alignment=1) has higher favor probability
        odds_ratio = np.exp(row['beta1'])
        return f"Opposition {odds_ratio:.2f}x more likely to favor"
    else:
        # Negative β1 means pro-ruling (alignment=0) has higher favor probability
        odds_ratio = np.exp(-row['beta1'])
        return f"Pro-ruling {odds_ratio:.2f}x more likely to favor"


def run_significance_tests(df):
    """Run binomial GLM for each keyword/aspect."""
    print("\nRunning significance tests per aspect...")
    
    keywords = df['keyword'].unique()
    results = []
    
    for keyword in keywords:
        print(f"  Processing: {keyword}")
        
        # Aggregate data by influencer
        influencer_data = aggregate_by_influencer(df, keyword)
        
        if influencer_data is None or len(influencer_data) < 5:
            print(f"    Skipped (insufficient data)")
            continue
        
        # Fit binomial GLM
        model_result = fit_binomial_glm(influencer_data)
        
        if model_result is None:
            print(f"    Skipped (model fitting failed)")
            continue
        
        results.append({
            'keyword': keyword,
            'beta1': model_result['beta1'],
            'std_err': model_result['std_err'],
            'z_value': model_result['z_value'],
            'p_value': model_result['p_value'],
            'significant': model_result['p_value'] < 0.05,
            'n_influencers': model_result['n_influencers'],
            'n_pro_ruling_influencers': model_result['n_pro_ruling_influencers'],
            'n_opposition_influencers': model_result['n_opposition_influencers'],
            'total_tweets': model_result['total_tweets'],
            'total_favor_tweets': model_result['total_favor']
        })
    
    results_df = pd.DataFrame(results)
    
    # Add interpretation
    if len(results_df) > 0:
        results_df['interpretation'] = results_df.apply(interpret_result, axis=1)
        results_df['odds_ratio'] = np.exp(results_df['beta1'])
    
    return results_df


def generate_summary_report(results_df):
    """Generate a summary report."""
    lines = []
    lines.append("=" * 80)
    lines.append("BINOMIAL GLM SIGNIFICANCE TESTING SUMMARY")
    lines.append("Model: logit(p_i) = β0 + β1 * Alignment_i + u_i")
    lines.append("=" * 80)
    lines.append("")
    
    # Overview
    n_sig = results_df['significant'].sum()
    n_total = len(results_df)
    lines.append("OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"Total aspects tested: {n_total}")
    lines.append(f"Significant results (p<0.05): {n_sig} ({n_sig/n_total*100:.1f}%)")
    lines.append("")
    
    # Significant findings
    lines.append("=" * 80)
    lines.append("SIGNIFICANT FINDINGS (p < 0.05)")
    lines.append("-" * 40)
    
    sig_results = results_df[results_df['significant']].sort_values('p_value')
    
    if len(sig_results) > 0:
        for _, row in sig_results.iterrows():
            lines.append(f"\n  {row['keyword']}:")
            lines.append(f"    β1 = {row['beta1']:.4f} (SE: {row['std_err']:.4f})")
            lines.append(f"    z = {row['z_value']:.3f}, p = {row['p_value']:.2e}")
            lines.append(f"    Odds Ratio = {row['odds_ratio']:.3f}")
            lines.append(f"    Interpretation: {row['interpretation']}")
            lines.append(f"    Influencers: {row['n_influencers']} ({row['n_pro_ruling_influencers']} pro-ruling, {row['n_opposition_influencers']} opposition)")
            lines.append(f"    Tweets: {row['total_tweets']} total, {row['total_favor_tweets']} favor")
    else:
        lines.append("  No significant results found.")
    
    # Non-significant (top 5)
    lines.append("")
    lines.append("=" * 80)
    lines.append("NON-SIGNIFICANT RESULTS (Top 5 by p-value)")
    lines.append("-" * 40)
    
    nonsig = results_df[~results_df['significant']].sort_values('p_value').head(5)
    for _, row in nonsig.iterrows():
        lines.append(f"  {row['keyword']}: β1={row['beta1']:.4f}, p={row['p_value']:.3f}")
    
    # Methodology
    lines.append("")
    lines.append("=" * 80)
    lines.append("METHODOLOGY")
    lines.append("-" * 40)
    lines.append("- Model: Binomial GLM with logit link")
    lines.append("- Response: [n_favor, n_total - n_favor] per influencer")
    lines.append("- Predictor: Alignment (0=pro-ruling, 1=opposition)")
    lines.append("- β1 > 0: Opposition influencers more likely to tweet 'favor'")
    lines.append("- β1 < 0: Pro-ruling influencers more likely to tweet 'favor'")
    lines.append("- Significance threshold: p < 0.05")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def main():
    """Main execution."""
    print("=" * 60)
    print("BINOMIAL GLM SIGNIFICANCE TESTING")
    print("Model: logit(p_i) = β0 + β1 * Alignment_i + u_i")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    df = load_data()
    
    # Run tests
    results_df = run_significance_tests(df)
    
    # Save results
    print("\nSaving results...")
    
    results_path = os.path.join(OUTPUT_DIR, "significance_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"  Saved: significance_results.csv")
    
    # Generate and save summary
    summary = generate_summary_report(results_df)
    summary_path = os.path.join(OUTPUT_DIR, "summary_report.txt")
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"  Saved: summary_report.txt")
    
    # Print summary
    print("\n" + summary)
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
