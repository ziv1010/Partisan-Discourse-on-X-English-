#!/usr/bin/env python3
"""
Significance Testing for Stance Analysis (Logistic Regression)

Uses logistic regression to test if political leaning significantly predicts
stance (favor/against/neutral) for each keyword.

The coefficient's p-value indicates whether the difference is significant.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import warnings
import os

warnings.filterwarnings('ignore')

# File paths
ENGLISH_DATA = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/final_results+visualisations_folder/stance_results_37keywords.csv"
HINDI_DATA = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/final_results+visualisations_folder/hindi_stance_results.csv"
OUTPUT_DIR = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/last minute work/significance_testing"


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


def logistic_regression_test(df, keyword, stance):
    """
    Use logistic regression to test if political leaning predicts stance.
    
    Model: P(stance=1) = logistic(β0 + β1*political_leaning)
    
    Where political_leaning = 1 for pro-ruling, 0 for pro-opposition.
    
    Returns: (coefficient, odds_ratio, p_value, significant, n_ruling, n_opp)
    """
    try:
        mask = df['keyword'] == keyword
        subset = df[mask].copy()
        
        if len(subset) < 20:
            return np.nan, np.nan, np.nan, False, 0, 0
        
        stance_col = 'fewshot_label_for_against'
        
        # Create binary outcome: 1 if has stance, 0 otherwise
        subset['y'] = (subset[stance_col] == stance).astype(int)
        
        # Create binary predictor: 1 for pro-ruling, 0 for pro-opposition
        subset['x'] = (subset['_label_norm'] == 'pro ruling').astype(int)
        
        n_ruling = len(subset[subset['x'] == 1])
        n_opp = len(subset[subset['x'] == 0])
        
        if n_ruling < 10 or n_opp < 10:
            return np.nan, np.nan, np.nan, False, n_ruling, n_opp
        
        # Check if there's variation in both X and Y
        if subset['y'].nunique() < 2 or subset['x'].nunique() < 2:
            return np.nan, np.nan, np.nan, False, n_ruling, n_opp
        
        # Fit logistic regression using statsmodels for p-values
        X = sm.add_constant(subset['x'])
        y = subset['y']
        
        model = sm.Logit(y, X).fit(disp=0, method='bfgs', maxiter=100)
        
        # Get coefficient, odds ratio, and p-value for political leaning
        coef = model.params.get('x', np.nan)
        p_value = model.pvalues.get('x', np.nan)
        odds_ratio = np.exp(coef) if not np.isnan(coef) else np.nan
        
        return coef, odds_ratio, p_value, p_value < 0.05, n_ruling, n_opp
        
    except Exception as e:
        return np.nan, np.nan, np.nan, False, 0, 0


def logistic_regression_inter(df_english, df_hindi, keyword, political_leaning, stance):
    """
    Use logistic regression to test if language predicts stance.
    
    Model: P(stance=1) = logistic(β0 + β1*language)
    
    Where language = 1 for English, 0 for Hindi.
    """
    try:
        mask_en = (df_english['keyword'] == keyword) & (df_english['_label_norm'] == political_leaning)
        mask_hi = (df_hindi['keyword'] == keyword) & (df_hindi['_label_norm'] == political_leaning)
        
        subset_en = df_english[mask_en].copy()
        subset_hi = df_hindi[mask_hi].copy()
        
        if len(subset_en) < 10 or len(subset_hi) < 10:
            return np.nan, np.nan, np.nan, False, len(subset_en), len(subset_hi)
        
        stance_col = 'fewshot_label_for_against'
        
        # Combine datasets
        subset_en['y'] = (subset_en[stance_col] == stance).astype(int)
        subset_en['x'] = 1  # English = 1
        
        subset_hi['y'] = (subset_hi[stance_col] == stance).astype(int)
        subset_hi['x'] = 0  # Hindi = 0
        
        combined = pd.concat([subset_en[['y', 'x']], subset_hi[['y', 'x']]], ignore_index=True)
        
        if combined['y'].nunique() < 2:
            return np.nan, np.nan, np.nan, False, len(subset_en), len(subset_hi)
        
        X = sm.add_constant(combined['x'])
        y = combined['y']
        
        model = sm.Logit(y, X).fit(disp=0, method='bfgs', maxiter=100)
        
        coef = model.params.get('x', np.nan)
        p_value = model.pvalues.get('x', np.nan)
        odds_ratio = np.exp(coef) if not np.isnan(coef) else np.nan
        
        return coef, odds_ratio, p_value, p_value < 0.05, len(subset_en), len(subset_hi)
        
    except Exception:
        return np.nan, np.nan, np.nan, False, 0, 0


def run_intra_dataset_tests(df, language):
    """
    Run logistic regression tests for each keyword.
    Tests if political leaning predicts stance.
    """
    print(f"\nRunning logistic regression tests for {language}...")
    
    keywords = df['keyword'].unique()
    results = []
    
    for keyword in keywords:
        for stance in ['favor', 'against', 'neutral']:
            coef, odds_ratio, p_value, significant, n_ruling, n_opp = logistic_regression_test(df, keyword, stance)
            
            if n_ruling >= 10 and n_opp >= 10:
                results.append({
                    'keyword': keyword,
                    'stance': stance,
                    'test_method': 'logistic_regression',
                    'coefficient': coef,
                    'odds_ratio': odds_ratio,
                    'p_value': p_value,
                    'significant': significant,
                    'n_pro_ruling': n_ruling,
                    'n_pro_opposition': n_opp,
                    'interpretation': interpret_odds_ratio(odds_ratio, significant)
                })
    
    print(f"  Processed {len(keywords)} keywords, generated {len(results)} test results")
    return pd.DataFrame(results)


def run_inter_dataset_tests(df_english, df_hindi):
    """
    Run logistic regression tests comparing English vs Hindi.
    Tests if language predicts stance.
    """
    print("\nRunning inter-dataset logistic regression tests...")
    
    results = []
    
    english_keywords = set(df_english['keyword'].unique())
    hindi_keywords = set(df_hindi['keyword'].unique())
    common_keywords = english_keywords.intersection(hindi_keywords)
    
    print(f"  Found {len(common_keywords)} common keywords")
    
    for political_leaning in ['pro ruling', 'pro opposition']:
        for keyword in common_keywords:
            for stance in ['favor', 'against', 'neutral']:
                coef, odds_ratio, p_value, significant, n_en, n_hi = logistic_regression_inter(
                    df_english, df_hindi, keyword, political_leaning, stance
                )
                
                if n_en >= 10 and n_hi >= 10:
                    results.append({
                        'keyword': keyword,
                        'political_leaning': political_leaning,
                        'stance': stance,
                        'test_method': 'logistic_regression',
                        'coefficient': coef,
                        'odds_ratio': odds_ratio,
                        'p_value': p_value,
                        'significant': significant,
                        'n_english': n_en,
                        'n_hindi': n_hi,
                        'interpretation': interpret_odds_ratio(odds_ratio, significant, "English vs Hindi")
                    })
    
    print(f"  Generated {len(results)} inter-dataset test results")
    return pd.DataFrame(results)


def interpret_odds_ratio(odds_ratio, significant, comparison="Pro-Ruling vs Pro-Opp"):
    """Interpret the odds ratio in plain language."""
    if np.isnan(odds_ratio) or not significant:
        return "No significant difference"
    
    if comparison == "Pro-Ruling vs Pro-Opp":
        if odds_ratio > 1:
            return f"Pro-ruling {odds_ratio:.1f}x more likely"
        else:
            return f"Pro-opp {1/odds_ratio:.1f}x more likely"
    else:
        if odds_ratio > 1:
            return f"English {odds_ratio:.1f}x more likely"
        else:
            return f"Hindi {1/odds_ratio:.1f}x more likely"


def generate_summary_report(intra_english_df, intra_hindi_df, inter_df):
    """Generate summary report."""
    
    lines = []
    lines.append("=" * 80)
    lines.append("LOGISTIC REGRESSION SIGNIFICANCE TESTING SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    
    lines.append("OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"English: {len(intra_english_df)} tests, {intra_english_df['significant'].sum()} significant ({intra_english_df['significant'].mean()*100:.1f}%)")
    lines.append(f"Hindi: {len(intra_hindi_df)} tests, {intra_hindi_df['significant'].sum()} significant ({intra_hindi_df['significant'].mean()*100:.1f}%)")
    lines.append(f"Inter-dataset: {len(inter_df)} tests, {inter_df['significant'].sum()} significant ({inter_df['significant'].mean()*100:.1f}%)")
    lines.append("")
    
    # English significant findings with odds ratios
    lines.append("=" * 80)
    lines.append("ENGLISH: Significant Findings (by Odds Ratio)")
    lines.append("-" * 40)
    
    for stance in ['favor', 'against', 'neutral']:
        sig = intra_english_df[(intra_english_df['significant'] == True) & (intra_english_df['stance'] == stance)]
        sig = sig.sort_values('odds_ratio', key=lambda x: abs(np.log(x)), ascending=False)
        lines.append(f"\n  {stance.upper()} stance: {len(sig)} significant keywords")
        for _, row in sig.head(10).iterrows():
            lines.append(f"    {row['keyword']}: OR={row['odds_ratio']:.2f}, p={row['p_value']:.2e} ({row['interpretation']})")
    
    # Hindi
    lines.append("")
    lines.append("=" * 80)
    lines.append("HINDI: Significant Findings (by Odds Ratio)")
    lines.append("-" * 40)
    
    for stance in ['favor', 'against', 'neutral']:
        sig = intra_hindi_df[(intra_hindi_df['significant'] == True) & (intra_hindi_df['stance'] == stance)]
        sig = sig.sort_values('odds_ratio', key=lambda x: abs(np.log(x)), ascending=False)
        lines.append(f"\n  {stance.upper()} stance: {len(sig)} significant keywords")
        for _, row in sig.head(10).iterrows():
            lines.append(f"    {row['keyword']}: OR={row['odds_ratio']:.2f}, p={row['p_value']:.2e} ({row['interpretation']})")
    
    # Methodology
    lines.append("")
    lines.append("=" * 80)
    lines.append("METHODOLOGY")
    lines.append("-" * 40)
    lines.append("- Test: Logistic Regression")
    lines.append("- Model: P(stance=1) = logistic(β0 + β1 * predictor)")
    lines.append("- Odds Ratio (OR): exp(β1)")
    lines.append("  - OR > 1: First group more likely to have that stance")
    lines.append("  - OR < 1: Second group more likely")
    lines.append("- Significance threshold: p < 0.05")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def main():
    """Main execution."""
    print("=" * 60)
    print("LOGISTIC REGRESSION SIGNIFICANCE TESTING")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df_english, df_hindi = load_data()
    
    # Run tests
    intra_english_df = run_intra_dataset_tests(df_english, 'English')
    intra_hindi_df = run_intra_dataset_tests(df_hindi, 'Hindi')
    inter_df = run_inter_dataset_tests(df_english, df_hindi)
    
    # Save results
    print("\nSaving results...")
    
    intra_english_df.to_csv(os.path.join(OUTPUT_DIR, "intra_english_results.csv"), index=False)
    print(f"  Saved: intra_english_results.csv")
    
    intra_hindi_df.to_csv(os.path.join(OUTPUT_DIR, "intra_hindi_results.csv"), index=False)
    print(f"  Saved: intra_hindi_results.csv")
    
    inter_df.to_csv(os.path.join(OUTPUT_DIR, "inter_dataset_results.csv"), index=False)
    print(f"  Saved: inter_dataset_results.csv")
    
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
