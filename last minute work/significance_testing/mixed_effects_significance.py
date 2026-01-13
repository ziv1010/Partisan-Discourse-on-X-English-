#!/usr/bin/env python3
"""
Significance Testing - Mixed Effects Logistic Regression

Uses Mixed Effects Logistic Regression to test if influencer alignment 
significantly predicts the probability of "favor" tweets for each aspect.

Model: logit(p_ij) = β0 + β1 * Alignment_i + u_i
- p_ij: probability of favor for tweet j from influencer i
- Alignment_i: 0 for pro-ruling, 1 for opposition
- u_i: random intercept for influencer i (accounts for influencer-level clustering)

This uses statsmodels BinomialBayesMixedGLM for mixed effects modeling.
Analysis is done separately per aspect (keyword).
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
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


def prepare_data_for_keyword(df, keyword):
    """
    Prepare data for mixed effects logistic regression.
    
    Each tweet is one observation.
    Response: 1 if favor, 0 otherwise
    Fixed effect: Alignment (0=pro-ruling, 1=opposition)
    Random effect: Influencer (random intercept)
    """
    # Filter for this keyword
    subset = df[df['keyword'] == keyword].copy()
    
    if len(subset) < 50:
        return None
    
    # Create binary response: 1 if favor, 0 otherwise
    subset['y'] = (subset['fewshot_label_for_against'] == 'favor').astype(int)
    
    # Create binary alignment: 0 for pro-ruling, 1 for opposition
    subset['alignment'] = (subset['_label_norm'] == 'pro opposition').astype(int)
    
    # Create influencer ID for grouping (random effect)
    # Need numeric IDs for the mixed model
    subset['influencer_id'] = pd.Categorical(subset['original_author']).codes
    
    # Count influencers and check variation
    n_influencers = subset['original_author'].nunique()
    n_ruling = (subset.groupby('original_author')['alignment'].first() == 0).sum()
    n_opp = (subset.groupby('original_author')['alignment'].first() == 1).sum()
    
    if n_ruling < 3 or n_opp < 3:
        return None
    
    # Check for variation in outcome
    if subset['y'].nunique() < 2:
        return None
    
    return {
        'data': subset,
        'n_tweets': len(subset),
        'n_influencers': n_influencers,
        'n_ruling_influencers': n_ruling,
        'n_opp_influencers': n_opp,
        'n_favor': subset['y'].sum()
    }


def fit_mixed_effects_model(prepared_data):
    """
    Fit Mixed Effects Logistic Regression using BinomialBayesMixedGLM.
    
    Model: logit(p_ij) = β0 + β1 * Alignment_i + u_i
    Where u_i ~ N(0, σ²) is a random intercept for influencer i
    
    Returns model results or None if fitting fails.
    """
    try:
        data = prepared_data['data']
        
        # Response variable (binary: 1/0)
        endog = data['y'].values
        
        # Fixed effects: intercept + alignment
        exog = sm.add_constant(data['alignment'].values)
        
        # Random effects grouping: influencer
        # Each influencer gets their own random intercept
        exog_vc = data['influencer_id'].values.reshape(-1, 1)
        
        # Group identifiers for random effects
        ident = np.zeros(1, dtype=int)  # Single random effect type
        
        # Fit the Bayesian Mixed GLM for binomial data
        model = BinomialBayesMixedGLM(
            endog=endog,
            exog=exog,
            exog_vc=exog_vc,
            ident=ident,
            vcp_p=0.5  # Prior SD for variance components
        )
        
        # Fit using Laplace approximation (maximum a posteriori)
        result = model.fit_map(method='BFGS', minim_opts={'maxiter': 200})
        
        # Extract fixed effects
        # params[0] = intercept (β0)
        # params[1] = alignment coefficient (β1)
        beta0 = result.fe_mean[0]
        beta1 = result.fe_mean[1]
        
        # Standard errors for fixed effects
        se_beta0 = result.fe_sd[0]
        se_beta1 = result.fe_sd[1]
        
        # Calculate z-value and p-value for β1
        z_value = beta1 / se_beta1
        # Two-tailed p-value from normal distribution
        from scipy import stats
        p_value = 2 * (1 - stats.norm.cdf(abs(z_value)))
        
        # Variance component (random effect variance)
        vc_sd = result.vcp_mean[0] if len(result.vcp_mean) > 0 else np.nan
        
        return {
            'beta0': beta0,
            'beta1': beta1,
            'se_beta0': se_beta0,
            'se_beta1': se_beta1,
            'z_value': z_value,
            'p_value': p_value,
            'random_effect_sd': vc_sd,
            'converged': True
        }
        
    except Exception as e:
        print(f"    Error fitting mixed model: {e}")
        return None


def interpret_result(row):
    """Interpret the β1 coefficient."""
    if pd.isna(row['p_value']) or not row['significant']:
        return "No significant difference"
    
    if row['beta1'] > 0:
        odds_ratio = np.exp(row['beta1'])
        return f"Opposition {odds_ratio:.2f}x more likely to favor"
    else:
        odds_ratio = np.exp(-row['beta1'])
        return f"Pro-ruling {odds_ratio:.2f}x more likely to favor"


def run_mixed_effects_tests(df):
    """Run mixed effects logistic regression for each keyword/aspect."""
    print("\nRunning Mixed Effects Logistic Regression per aspect...")
    print("(This may take a while due to Bayesian estimation)")
    
    keywords = df['keyword'].unique()
    results = []
    
    for i, keyword in enumerate(keywords):
        print(f"  [{i+1}/{len(keywords)}] Processing: {keyword}", end="", flush=True)
        
        # Prepare data
        prepared = prepare_data_for_keyword(df, keyword)
        
        if prepared is None:
            print(" - Skipped (insufficient data)")
            continue
        
        # Fit mixed effects model
        model_result = fit_mixed_effects_model(prepared)
        
        if model_result is None:
            print(" - Skipped (model fitting failed)")
            continue
        
        results.append({
            'keyword': keyword,
            'beta0': model_result['beta0'],
            'beta1': model_result['beta1'],
            'se_beta0': model_result['se_beta0'],
            'se_beta1': model_result['se_beta1'],
            'z_value': model_result['z_value'],
            'p_value': model_result['p_value'],
            'significant': model_result['p_value'] < 0.05,
            'random_effect_sd': model_result['random_effect_sd'],
            'n_tweets': prepared['n_tweets'],
            'n_influencers': prepared['n_influencers'],
            'n_pro_ruling_influencers': prepared['n_ruling_influencers'],
            'n_opposition_influencers': prepared['n_opp_influencers'],
            'n_favor_tweets': prepared['n_favor']
        })
        
        sig_marker = "**" if model_result['p_value'] < 0.05 else ""
        print(f" - Done (β1={model_result['beta1']:.3f}, p={model_result['p_value']:.3e}) {sig_marker}")
    
    results_df = pd.DataFrame(results)
    
    # Add interpretation and odds ratio
    if len(results_df) > 0:
        results_df['interpretation'] = results_df.apply(interpret_result, axis=1)
        results_df['odds_ratio'] = np.exp(results_df['beta1'])
    
    return results_df


def generate_summary_report(results_df):
    """Generate a summary report."""
    lines = []
    lines.append("=" * 80)
    lines.append("MIXED EFFECTS LOGISTIC REGRESSION SIGNIFICANCE TESTING")
    lines.append("Model: logit(p_ij) = β0 + β1 * Alignment_i + u_i")
    lines.append("Where u_i ~ N(0, σ²) is a random intercept for influencer i")
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
            lines.append(f"    β1 = {row['beta1']:.4f} (SE: {row['se_beta1']:.4f})")
            lines.append(f"    z = {row['z_value']:.3f}, p = {row['p_value']:.2e}")
            lines.append(f"    Odds Ratio = {row['odds_ratio']:.3f}")
            lines.append(f"    Random Effect SD = {row['random_effect_sd']:.4f}")
            lines.append(f"    Interpretation: {row['interpretation']}")
            lines.append(f"    Tweets: {row['n_tweets']} from {row['n_influencers']} influencers")
            lines.append(f"    Influencer split: {row['n_pro_ruling_influencers']} pro-ruling, {row['n_opposition_influencers']} opposition")
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
    lines.append("- Model: Mixed Effects Logistic Regression (BinomialBayesMixedGLM)")
    lines.append("- Fixed Effect: Alignment (0=pro-ruling, 1=opposition)")
    lines.append("- Random Effect: Random intercept per influencer u_i ~ N(0, σ²)")
    lines.append("- Each tweet is one observation (not aggregated)")
    lines.append("- Estimation: Bayesian MAP via Laplace approximation")
    lines.append("- β1 > 0: Opposition influencers more likely to tweet 'favor'")
    lines.append("- β1 < 0: Pro-ruling influencers more likely to tweet 'favor'")
    lines.append("- Significance threshold: p < 0.05")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def main():
    """Main execution."""
    print("=" * 60)
    print("MIXED EFFECTS LOGISTIC REGRESSION SIGNIFICANCE TESTING")
    print("Model: logit(p_ij) = β0 + β1 * Alignment_i + u_i")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    df = load_data()
    
    # Run tests
    results_df = run_mixed_effects_tests(df)
    
    # Save results
    print("\nSaving results...")
    
    results_path = os.path.join(OUTPUT_DIR, "mixed_effects_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"  Saved: mixed_effects_results.csv")
    
    # Generate and save summary
    summary = generate_summary_report(results_df)
    summary_path = os.path.join(OUTPUT_DIR, "mixed_effects_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"  Saved: mixed_effects_summary.txt")
    
    # Print summary
    print("\n" + summary)
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
