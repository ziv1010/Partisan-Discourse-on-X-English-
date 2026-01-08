#!/usr/bin/env python3
"""
Per-Keyword Analysis Script
Generates detailed per-keyword accuracy and F1 scores for all 5 models.
Saves results to CSV files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
import argparse

# Paths
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"

# Model configurations
MODEL_CONFIGS = {
    'bert': {'file': 'bert_predictions.csv', 'gt_col': 'original_stance', 'pred_col': 'bert_prediction'},
    'roberta': {'file': 'roberta_predictions.csv', 'gt_col': 'original_stance', 'pred_col': 'roberta_prediction'},
    'mistral_base': {'file': 'mistral_base_predictions.csv', 'gt_col': 'original_stance', 'pred_col': 'mistral_prediction'},
    'mistral_fewshot': {'file': 'mistral_fewshot_predictions.csv', 'gt_col': 'stance', 'pred_col': 'fewshot_label'},
    'mistral_lora': {'file': 'mistral_lora_predictions.csv', 'gt_col': 'stance', 'pred_col': 'fewshot_label'},
}

LABELS = ['For', 'Against', 'Neutral']


def normalize_stance(s):
    """Normalize stance labels."""
    if pd.isna(s):
        return None
    s = str(s).lower().strip()
    s = s.replace('favour', 'for').replace('favor', 'for').replace('nuetral', 'neutral')
    if s in ['for', 'positive']:
        return 'For'
    elif s in ['against', 'negative']:
        return 'Against'
    elif s in ['neutral']:
        return 'Neutral'
    return None


def analyze_model(model_name: str, config: dict, results_dir: Path) -> pd.DataFrame:
    """Analyze per-keyword performance for a single model."""
    pred_path = results_dir / config['file']
    
    if not pred_path.exists():
        print(f"  ✗ Not found: {pred_path}")
        return None
    
    df = pd.read_csv(pred_path)
    
    # Find ground truth column
    gt_col = config['gt_col'] if config['gt_col'] in df.columns else 'original_stance'
    if gt_col not in df.columns and 'stance' in df.columns:
        gt_col = 'stance'
    
    # Find prediction column
    pred_col = config['pred_col']
    if pred_col not in df.columns:
        print(f"  ✗ Prediction column '{pred_col}' not found in {model_name}")
        return None
    
    # Normalize stances
    df['gt'] = df[gt_col].apply(normalize_stance)
    df['pred'] = df[pred_col].apply(normalize_stance)
    
    # Filter valid samples
    valid = df[df['gt'].notna() & df['pred'].notna()]
    
    results = []
    for kw in sorted(valid['keyword'].unique()):
        kw_df = valid[valid['keyword'] == kw]
        gt = kw_df['gt'].tolist()
        pred = kw_df['pred'].tolist()
        
        correct = sum(1 for g, p in zip(gt, pred) if g == p)
        acc = correct / len(kw_df) if len(kw_df) > 0 else 0
        f1 = f1_score(gt, pred, average='macro', labels=LABELS, zero_division=0)
        
        # Per-class counts
        for_gt = (kw_df['gt'] == 'For').sum()
        against_gt = (kw_df['gt'] == 'Against').sum()
        neutral_gt = (kw_df['gt'] == 'Neutral').sum()
        
        results.append({
            'model': model_name,
            'keyword': kw,
            'samples': len(kw_df),
            'correct': correct,
            'accuracy': round(acc * 100, 2),
            'f1_macro': round(f1, 4),
            'gt_for': for_gt,
            'gt_against': against_gt,
            'gt_neutral': neutral_gt,
        })
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Per-Keyword Analysis')
    parser.add_argument('--results-dir', type=str, default=str(RESULTS_DIR), help='Results directory')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    print("=" * 80)
    print("Per-Keyword Analysis for All Models")
    print("=" * 80)
    
    all_results = []
    
    for model_name, config in MODEL_CONFIGS.items():
        print(f"\nProcessing {model_name}...")
        model_results = analyze_model(model_name, config, results_dir)
        
        if model_results is not None:
            all_results.append(model_results)
            print(f"  ✓ Analyzed {len(model_results)} keywords")
    
    if not all_results:
        print("No results to process!")
        return
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save combined per-keyword results
    output_path = results_dir / "per_keyword_detailed.csv"
    combined_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved detailed per-keyword results to: {output_path}")
    
    # Create pivot table for easier comparison
    pivot_acc = combined_df.pivot(index='keyword', columns='model', values='accuracy')
    pivot_acc = pivot_acc.reindex(columns=['bert', 'roberta', 'mistral_base', 'mistral_fewshot', 'mistral_lora'])
    pivot_acc_path = results_dir / "per_keyword_accuracy_comparison.csv"
    pivot_acc.to_csv(pivot_acc_path)
    print(f"✓ Saved accuracy comparison to: {pivot_acc_path}")
    
    pivot_f1 = combined_df.pivot(index='keyword', columns='model', values='f1_macro')
    pivot_f1 = pivot_f1.reindex(columns=['bert', 'roberta', 'mistral_base', 'mistral_fewshot', 'mistral_lora'])
    pivot_f1_path = results_dir / "per_keyword_f1_comparison.csv"
    pivot_f1.to_csv(pivot_f1_path)
    print(f"✓ Saved F1 comparison to: {pivot_f1_path}")
    
    # Print summary tables
    print("\n" + "=" * 80)
    print("PER-KEYWORD ACCURACY (%) BY MODEL")
    print("=" * 80)
    print(pivot_acc.round(1).to_string())
    
    print("\n" + "=" * 80)
    print("PER-KEYWORD F1 (MACRO) BY MODEL")
    print("=" * 80)
    print(pivot_f1.round(3).to_string())
    
    # Per-keyword sample counts (from any model, they should be same)
    first_model = combined_df[combined_df['model'] == 'mistral_lora'][['keyword', 'samples', 'gt_for', 'gt_against', 'gt_neutral']]
    first_model = first_model.set_index('keyword')
    
    print("\n" + "=" * 80)
    print("ANNOTATION COUNTS PER KEYWORD")
    print("=" * 80)
    print(first_model.to_string())
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
