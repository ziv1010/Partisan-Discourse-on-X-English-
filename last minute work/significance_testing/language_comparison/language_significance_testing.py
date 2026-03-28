#!/usr/bin/env python3
"""
Significance Testing for Language Comparison (English vs Hindi)

Uses binomial GLM to test if language (English vs Hindi) significantly predicts
the count of "favor" tweets for each aspect (keyword).

Model: logit(p) = β0 + β1 * Language
- p: probability of favor
- Language: 0 for English, 1 for Hindi

Analysis is done separately per aspect (keyword) - 35 tests total.
If β1 ≠ 0 and p < 0.05, language significantly affects favor tweet counts.

Goal: Show that differences between English and Hindi are NOT significant.
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
OUTPUT_DIR = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/last minute work/significance_testing/language_comparison"


def load_data():
    """Load the combined dataset."""
    print("Loading dataset...")
    df = pd.read_csv(COMBINED_DATA, low_memory=False)
    print(f"  Total rows: {len(df):,}")
    print(f"  Unique keywords: {df['keyword'].nunique()}")
    print(f"  English tweets: {(df['language'] == 'english').sum():,}")
    print(f"  Hindi tweets: {(df['language'] == 'hindi').sum():,}")
    return df


def aggregate_by_language(df, keyword):
    """
    For a given keyword, aggregate data by language.
    
    Returns DataFrame with:
    - language: 'english' or 'hindi'
    - language_code: 0 for English, 1 for Hindi
    - n_favor: count of favor tweets
    - n_total: total tweet count
    """
    # Filter for this keyword
    subset = df[df['keyword'] == keyword].copy()
    
    if len(subset) == 0:
        return None
    
    # Group by language
    grouped = subset.groupby('language').agg({
        'fewshot_label_for_against': lambda x: (x == 'favor').sum(),  # n_favor
        'keyword': 'count'  # n_total
    }).reset_index()
    
    grouped.columns = ['language', 'n_favor', 'n_total']
    
    # Encode language: 0 for English, 1 for Hindi
    grouped['language_code'] = (grouped['language'] == 'hindi').astype(int)
    
    # Calculate n_not_favor for binomial response
    grouped['n_not_favor'] = grouped['n_total'] - grouped['n_favor']
    
    # Calculate favor percentage
    grouped['favor_pct'] = grouped['n_favor'] / grouped['n_total'] * 100
    
    return grouped


def fit_binomial_glm(lang_data):
    """
    Fit binomial GLM: logit(p) = β0 + β1 * Language
    
    Returns: dict with model results or None
    """
    try:
        # Need both languages present
        if len(lang_data) < 2:
            return None
        
        # Check for variation in outcome
        if lang_data['n_favor'].sum() == 0 or lang_data['n_favor'].sum() == lang_data['n_total'].sum():
            return None
        
        english_data = lang_data[lang_data['language'] == 'english']
        hindi_data = lang_data[lang_data['language'] == 'hindi']
        
        n_english = english_data['n_total'].values[0] if len(english_data) > 0 else 0
        n_hindi = hindi_data['n_total'].values[0] if len(hindi_data) > 0 else 0
        
        if n_english < 10 or n_hindi < 10:
            return None
        
        # Prepare response variable (success, failure counts)
        endog = lang_data[['n_favor', 'n_not_favor']].values
        
        # Prepare predictor with constant
        exog = sm.add_constant(lang_data['language_code'])
        
        # Fit binomial GLM
        model = sm.GLM(endog, exog, family=Binomial())
        result = model.fit(disp=0)
        
        # Extract β1 coefficient (for language)
        beta1 = result.params.iloc[1]  # Language coefficient
        std_err = result.bse.iloc[1]
        z_value = result.tvalues.iloc[1]
        p_value = result.pvalues.iloc[1]
        
        english_favor_pct = english_data['favor_pct'].values[0] if len(english_data) > 0 else np.nan
        hindi_favor_pct = hindi_data['favor_pct'].values[0] if len(hindi_data) > 0 else np.nan
        
        return {
            'beta1': beta1,
            'std_err': std_err,
            'z_value': z_value,
            'p_value': p_value,
            'n_english_tweets': n_english,
            'n_hindi_tweets': n_hindi,
            'total_tweets': lang_data['n_total'].sum(),
            'total_favor': lang_data['n_favor'].sum(),
            'english_favor_pct': english_favor_pct,
            'hindi_favor_pct': hindi_favor_pct
        }
        
    except Exception as e:
        print(f"    Error fitting model: {e}")
        return None


def interpret_result(row):
    """Interpret the β1 coefficient for language comparison."""
    if pd.isna(row['p_value']):
        return "Model fitting failed"
    
    if not row['significant']:
        return "No significant difference between languages"
    
    if row['beta1'] > 0:
        odds_ratio = np.exp(row['beta1'])
        return f"Hindi {odds_ratio:.2f}x more likely to favor (SIGNIFICANT)"
    else:
        odds_ratio = np.exp(-row['beta1'])
        return f"English {odds_ratio:.2f}x more likely to favor (SIGNIFICANT)"


def run_significance_tests(df):
    """Run binomial GLM for each keyword/aspect comparing languages."""
    print("\nRunning significance tests per aspect (English vs Hindi)...")
    
    keywords = df['keyword'].unique()
    results = []
    
    for keyword in keywords:
        print(f"  Processing: {keyword}")
        
        # Aggregate data by language
        lang_data = aggregate_by_language(df, keyword)
        
        if lang_data is None or len(lang_data) < 2:
            print(f"    Skipped (insufficient data for both languages)")
            continue
        
        # Fit binomial GLM
        model_result = fit_binomial_glm(lang_data)
        
        if model_result is None:
            print(f"    Skipped (model fitting failed or insufficient data)")
            continue
        
        results.append({
            'keyword': keyword,
            'beta1': model_result['beta1'],
            'std_err': model_result['std_err'],
            'z_value': model_result['z_value'],
            'p_value': model_result['p_value'],
            'significant': model_result['p_value'] < 0.05,
            'n_english_tweets': model_result['n_english_tweets'],
            'n_hindi_tweets': model_result['n_hindi_tweets'],
            'total_tweets': model_result['total_tweets'],
            'total_favor_tweets': model_result['total_favor'],
            'english_favor_pct': model_result['english_favor_pct'],
            'hindi_favor_pct': model_result['hindi_favor_pct']
        })
    
    results_df = pd.DataFrame(results)
    
    # Add interpretation
    if len(results_df) > 0:
        results_df['interpretation'] = results_df.apply(interpret_result, axis=1)
        results_df['odds_ratio'] = np.exp(results_df['beta1'])
        results_df['favor_pct_diff'] = abs(results_df['hindi_favor_pct'] - results_df['english_favor_pct'])
    
    return results_df


def generate_summary_report(results_df):
    """Generate a summary report focused on showing language differences are not significant."""
    lines = []
    lines.append("=" * 80)
    lines.append("LANGUAGE COMPARISON: ENGLISH vs HINDI")
    lines.append("Testing if favor stance differs significantly between languages")
    lines.append("Model: logit(p) = β0 + β1 * Language (Language: 0=English, 1=Hindi)")
    lines.append("=" * 80)
    lines.append("")
    
    # Key Finding
    n_sig = results_df['significant'].sum()
    n_nonsig = (~results_df['significant']).sum()
    n_total = len(results_df)
    
    lines.append("★★★ KEY FINDING ★★★")
    lines.append("-" * 40)
    lines.append(f"Total aspects tested: {n_total}")
    lines.append(f"NON-SIGNIFICANT results (p >= 0.05): {n_nonsig} ({n_nonsig/n_total*100:.1f}%)")
    lines.append(f"Significant results (p < 0.05): {n_sig} ({n_sig/n_total*100:.1f}%)")
    lines.append("")
    lines.append(f"➤ For {n_nonsig/n_total*100:.1f}% of aspects, the difference between")
    lines.append(f"  English and Hindi favor stance is NOT statistically significant.")
    lines.append("")
    
    # Average favor percentages
    lines.append("AVERAGE FAVOR PERCENTAGES BY LANGUAGE")
    lines.append("-" * 40)
    avg_english = results_df['english_favor_pct'].mean()
    avg_hindi = results_df['hindi_favor_pct'].mean()
    avg_diff = results_df['favor_pct_diff'].mean()
    lines.append(f"Average English favor %: {avg_english:.2f}%")
    lines.append(f"Average Hindi favor %: {avg_hindi:.2f}%")
    lines.append(f"Average absolute difference: {avg_diff:.2f} percentage points")
    lines.append("")
    
    # Non-significant findings (prioritize these)
    lines.append("=" * 80)
    lines.append("NON-SIGNIFICANT RESULTS (No language difference)")
    lines.append("-" * 40)
    
    nonsig_results = results_df[~results_df['significant']].sort_values('p_value', ascending=False)
    
    if len(nonsig_results) > 0:
        for _, row in nonsig_results.iterrows():
            lines.append(f"\n  {row['keyword']}:")
            lines.append(f"    p-value = {row['p_value']:.4f} (NOT significant)")
            lines.append(f"    English favor: {row['english_favor_pct']:.1f}% (n={row['n_english_tweets']:,})")
            lines.append(f"    Hindi favor: {row['hindi_favor_pct']:.1f}% (n={row['n_hindi_tweets']:,})")
            lines.append(f"    Difference: {row['favor_pct_diff']:.1f} percentage points")
    else:
        lines.append("  No non-significant results found.")
    
    # Significant findings (if any)
    if n_sig > 0:
        lines.append("")
        lines.append("=" * 80)
        lines.append(f"SIGNIFICANT RESULTS (p < 0.05) - {n_sig} aspects")
        lines.append("-" * 40)
        
        sig_results = results_df[results_df['significant']].sort_values('p_value')
        
        for _, row in sig_results.iterrows():
            lines.append(f"\n  {row['keyword']}:")
            lines.append(f"    β1 = {row['beta1']:.4f} (SE: {row['std_err']:.4f})")
            lines.append(f"    z = {row['z_value']:.3f}, p = {row['p_value']:.2e}")
            lines.append(f"    English favor: {row['english_favor_pct']:.1f}% (n={row['n_english_tweets']:,})")
            lines.append(f"    Hindi favor: {row['hindi_favor_pct']:.1f}% (n={row['n_hindi_tweets']:,})")
            lines.append(f"    Interpretation: {row['interpretation']}")
    
    # Methodology
    lines.append("")
    lines.append("=" * 80)
    lines.append("METHODOLOGY")
    lines.append("-" * 40)
    lines.append("- Model: Binomial GLM with logit link")
    lines.append("- Response: [n_favor, n_total - n_favor] per language per aspect")
    lines.append("- Predictor: Language (0=English, 1=Hindi)")
    lines.append("- β1 > 0: Hindi has higher favor probability")
    lines.append("- β1 < 0: English has higher favor probability")
    lines.append("- Significance threshold: p < 0.05")
    lines.append("")
    lines.append("CONCLUSION:")
    lines.append(f"{n_nonsig}/{n_total} ({n_nonsig/n_total*100:.1f}%) of aspects show NO significant")
    lines.append("difference in favor stance between English and Hindi.")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def main():
    """Main execution."""
    print("=" * 60)
    print("LANGUAGE SIGNIFICANCE TESTING (English vs Hindi)")
    print("Model: logit(p) = β0 + β1 * Language")
    print("Testing if favor stance differs between languages")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    df = load_data()
    
    # Run tests
    results_df = run_significance_tests(df)
    
    # Save results
    print("\nSaving results...")
    
    results_path = os.path.join(OUTPUT_DIR, "language_significance_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"  Saved: language_significance_results.csv")
    
    # Generate and save summary
    summary = generate_summary_report(results_df)
    summary_path = os.path.join(OUTPUT_DIR, "language_comparison_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"  Saved: language_comparison_summary.txt")
    
    # Print summary
    print("\n" + summary)
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
