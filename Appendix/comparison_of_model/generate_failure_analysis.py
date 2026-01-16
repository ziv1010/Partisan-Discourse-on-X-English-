#!/usr/bin/env python3
"""
Failure Analysis Script
Generates CSV files for cases where models fail:
1. Cases where ALL models fail
2. Cases where majority of models fail for 'ram mandir' and 'hathras' keywords
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "failure_output"
RESULTS_DIR = Path("/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/4_finetuning/4_d_model_comparison_baseline/final_runs/results")

# Model configurations
MODEL_CONFIGS = {
    'bert': {'file': 'bert_predictions.csv', 'pred_col': 'bert_prediction', 'display': 'BERT'},
    'roberta': {'file': 'roberta_predictions.csv', 'pred_col': 'roberta_prediction', 'display': 'RoBERTa'},
    'pyabsa': {'file': 'pyabsa_predictions.csv', 'pred_col': 'pyabsa_prediction', 'display': 'PyABSA'},
    'mistral_base': {'file': 'mistral_base_predictions.csv', 'pred_col': 'mistral_prediction', 'display': 'Mistral (Zero-shot)'},
    'mistral_fewshot': {'file': 'mistral_fewshot_predictions.csv', 'pred_col': 'fewshot_label', 'display': 'Mistral (Few-shot)'},
    'mistral_lora': {'file': 'mistral_lora_predictions.csv', 'pred_col': 'fewshot_label', 'display': 'Mistral LoRA (Best)'},
}

LABELS = ["For", "Against", "Neutral"]


def normalize_stance(stance: str) -> str:
    """Normalize stance labels."""
    if pd.isna(stance):
        return None
    s = str(stance).lower().strip()
    s = s.replace('favour', 'for').replace('favor', 'for').replace('nuetral', 'neutral')
    if s in ['for', 'positive']:
        return 'For'
    elif s in ['against', 'negative']:
        return 'Against'
    elif s in ['neutral']:
        return 'Neutral'
    return None


def load_all_predictions():
    """Load predictions from all models."""
    predictions = {}
    
    for model_name, config in MODEL_CONFIGS.items():
        pred_path = RESULTS_DIR / config['file']
        if pred_path.exists():
            df = pd.read_csv(pred_path)
            
            # Normalize stance columns
            if 'original_stance' in df.columns:
                df['original_stance'] = df['original_stance'].apply(normalize_stance)
            elif 'stance' in df.columns:
                df['original_stance'] = df['stance'].apply(normalize_stance)
            
            if config['pred_col'] in df.columns:
                df['prediction'] = df[config['pred_col']].apply(normalize_stance)
            
            # Filter out rows with invalid stances
            valid_mask = df['original_stance'].notna() & df['prediction'].notna()
            df = df[valid_mask]
            
            predictions[model_name] = df
            print(f"✓ Loaded {model_name}: {len(df)} samples")
        else:
            print(f"✗ Not found: {pred_path}")
    
    return predictions


def create_merged_dataset(predictions):
    """Create a merged dataset with all model predictions."""
    # Use mistral_lora as base
    base_df = predictions.get('mistral_lora')
    if base_df is None:
        base_df = list(predictions.values())[0]
    
    merged = base_df[['tweet', 'keyword', 'original_stance']].copy()
    merged = merged.rename(columns={'original_stance': 'ground_truth'})
    
    for model_name, df in predictions.items():
        pred_df = df[['tweet', 'keyword', 'prediction']].copy()
        pred_df = pred_df.rename(columns={'prediction': f'{model_name}_pred'})
        merged = merged.merge(pred_df, on=['tweet', 'keyword'], how='left')
    
    return merged


def find_all_models_fail(merged, predictions):
    """Find cases where ALL models fail."""
    all_models = list(predictions.keys())
    
    merged['all_wrong'] = True
    merged['num_wrong'] = 0
    
    for model_name in all_models:
        col = f'{model_name}_pred'
        if col in merged.columns:
            is_wrong = merged[col] != merged['ground_truth']
            merged['all_wrong'] = merged['all_wrong'] & is_wrong
            merged['num_wrong'] = merged['num_wrong'] + is_wrong.astype(int)
    
    merged['total_models'] = len(all_models)
    
    # Cases where ALL models fail
    all_fail_cases = merged[merged['all_wrong']].copy()
    
    # Add common prediction column
    pred_cols = [f'{m}_pred' for m in all_models if f'{m}_pred' in merged.columns]
    
    def get_common_pred(row):
        preds = [row[col] for col in pred_cols if pd.notna(row[col])]
        if preds:
            return max(set(preds), key=preds.count)
        return None
    
    all_fail_cases['common_prediction'] = all_fail_cases.apply(get_common_pred, axis=1)
    
    return all_fail_cases, merged


def find_majority_fail_for_keywords(merged, predictions, keywords_of_interest):
    """Find cases where MAJORITY of models fail for specific keywords, including Mistral LoRA."""
    all_models = list(predictions.keys())
    num_models = len(all_models)
    majority_threshold = num_models // 2 + 1  # More than half
    
    # Filter for keywords of interest
    keyword_df = merged[merged['keyword'].str.lower().isin([k.lower() for k in keywords_of_interest])].copy()
    
    # Check if mistral_lora also fails
    keyword_df['mistral_lora_wrong'] = keyword_df['mistral_lora_pred'] != keyword_df['ground_truth']
    
    # Cases where majority fail AND mistral_lora also fails
    majority_fail = keyword_df[
        (keyword_df['num_wrong'] >= majority_threshold) & 
        (keyword_df['mistral_lora_wrong'] == True)
    ].copy()
    
    # Add common prediction column
    pred_cols = [f'{m}_pred' for m in all_models if f'{m}_pred' in merged.columns]
    
    def get_common_pred(row):
        preds = [row[col] for col in pred_cols if pd.notna(row[col])]
        if preds:
            return max(set(preds), key=preds.count)
        return None
    
    majority_fail['common_prediction'] = majority_fail.apply(get_common_pred, axis=1)
    
    return majority_fail


def main():
    print("=" * 60)
    print("Model Failure Analysis Generator")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load predictions
    print("\nLoading predictions...")
    predictions = load_all_predictions()
    
    if not predictions:
        print("ERROR: No prediction files found!")
        sys.exit(1)
    
    # Create merged dataset
    print("\nCreating merged dataset...")
    merged = create_merged_dataset(predictions)
    print(f"Total samples in merged dataset: {len(merged)}")
    
    # Find cases where ALL models fail
    print("\nFinding cases where ALL models fail...")
    all_fail_cases, merged_with_counts = find_all_models_fail(merged, predictions)
    print(f"Found {len(all_fail_cases)} cases where ALL models fail")
    
    # Select relevant columns for output
    model_pred_cols = [f'{m}_pred' for m in predictions.keys()]
    output_cols = ['tweet', 'keyword', 'ground_truth'] + model_pred_cols + ['common_prediction']
    
    # Save all fail cases
    if len(all_fail_cases) > 0:
        all_fail_output = all_fail_cases[output_cols]
        all_fail_path = OUTPUT_DIR / "all_models_fail.csv"
        all_fail_output.to_csv(all_fail_path, index=False)
        print(f"✓ Saved: {all_fail_path}")
        
        # Print some statistics
        print(f"\n--- All Models Fail Statistics ---")
        print(f"Total cases: {len(all_fail_cases)}")
        print(f"\nBy keyword:")
        print(all_fail_cases['keyword'].value_counts())
        print(f"\nBy ground truth:")
        print(all_fail_cases['ground_truth'].value_counts())
    
    # Find majority fail for ram mandir and hathras
    print("\n" + "=" * 60)
    print("Finding MAJORITY fail cases for 'ram mandir' and 'hathras'...")
    keywords_of_interest = ['ram mandir', 'hathras']
    
    majority_fail = find_majority_fail_for_keywords(merged_with_counts, predictions, keywords_of_interest)
    print(f"Found {len(majority_fail)} cases where MAJORITY of models fail for {keywords_of_interest}")
    
    # Save majority fail cases for ram mandir and hathras
    if len(majority_fail) > 0:
        extra_cols = ['num_wrong', 'total_models']
        output_cols_majority = ['tweet', 'keyword', 'ground_truth'] + model_pred_cols + ['num_wrong', 'total_models', 'common_prediction']
        majority_fail_output = majority_fail[output_cols_majority]
        majority_fail_path = OUTPUT_DIR / "majority_fail_ram_mandir_hathras.csv"
        majority_fail_output.to_csv(majority_fail_path, index=False)
        print(f"✓ Saved: {majority_fail_path}")
        
        # Print statistics
        print(f"\n--- Majority Fail Statistics (Ram Mandir & Hathras) ---")
        print(f"Total cases: {len(majority_fail)}")
        print(f"\nBy keyword:")
        print(majority_fail['keyword'].value_counts())
        print(f"\nBy ground truth:")
        print(majority_fail['ground_truth'].value_counts())
        print(f"\nBy number of models failing:")
        print(majority_fail['num_wrong'].value_counts().sort_index())
    
    # Also create a summary CSV
    summary_data = []
    for keyword in merged_with_counts['keyword'].unique():
        keyword_data = merged_with_counts[merged_with_counts['keyword'] == keyword]
        total = len(keyword_data)
        all_fail = len(keyword_data[keyword_data['all_wrong']])
        majority_fail_count = len(keyword_data[keyword_data['num_wrong'] >= len(predictions) // 2 + 1])
        
        summary_data.append({
            'keyword': keyword,
            'total_samples': total,
            'all_models_fail': all_fail,
            'majority_fail': majority_fail_count,
            'all_fail_pct': all_fail / total * 100 if total > 0 else 0,
            'majority_fail_pct': majority_fail_count / total * 100 if total > 0 else 0,
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('all_fail_pct', ascending=False)
    summary_path = OUTPUT_DIR / "failure_summary_by_keyword.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Saved summary: {summary_path}")
    
    print("\n" + "=" * 60)
    print("✓ All outputs generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
