#!/usr/bin/env python3
"""
Crosscheck Mistral LoRA Results against Values in Results.tex

This script verifies:
1. That tweets in mistral_lora_predictions.csv match master_test.csv
2. All metrics reported in the Detailed Evaluation Metrics section of Results.tex

Expected values from Results.tex:
- Total test samples: 264
- Accuracy: 78.0%
- Precision (macro): 77.6%
- Recall (macro): 75.3%
- F1-Score (macro): 76.1%
- F1-Score (weighted): 77.7%

Per-class (Against): Precision=0.78, Recall=0.80, F1=0.79, Support=95
Per-class (Favor): Precision=0.79, Recall=0.86, F1=0.82, Support=111
Per-class (Neutral): Precision=0.76, Recall=0.60, F1=0.67, Support=58

Per-keyword accuracy (Top 10):
- modi: 94.4%, F1=0.91
- caa: 89.5%, F1=0.90
- congress: 88.9%, F1=0.84
- hindutva: 85.7%, F1=0.84
- new_parliament: 85.7%, F1=0.82
- rahulgandhi: 81.2%, F1=0.82
- farmers_protests: 80.0%, F1=0.82
- muslim: 78.9%, F1=0.79
- shaheen_bagh: 78.9%, F1=0.71
- china: 73.7%, F1=0.64
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from pathlib import Path


def main():
    base_path = Path(__file__).parent.parent
    
    # File paths
    predictions_path = base_path / "results" / "mistral_lora_predictions.csv"
    master_test_path = base_path.parent.parent / "4_a_DataProcessing" / "data_formatting" / "master_test.csv"
    
    print("=" * 70)
    print("MISTRAL LoRA RESULTS CROSSCHECK")
    print("=" * 70)
    
    # Load data
    print("\n[1] LOADING DATA")
    print("-" * 40)
    predictions_df = pd.read_csv(predictions_path)
    master_test_df = pd.read_csv(master_test_path)
    
    print(f"Predictions file: {predictions_path}")
    print(f"  Rows: {len(predictions_df)}")
    print(f"  Columns: {list(predictions_df.columns)}")
    
    print(f"\nMaster test file: {master_test_path}")
    print(f"  Rows: {len(master_test_df)}")
    print(f"  Columns: {list(master_test_df.columns)}")
    
    # =========================================================================
    # VERIFICATION 1: Check tweets match between files
    # =========================================================================
    print("\n" + "=" * 70)
    print("[2] TWEET MATCHING VERIFICATION")
    print("-" * 40)
    
    # Check by source_row
    pred_source_rows = set(predictions_df['source_row'].dropna().astype(int))
    test_source_rows = set(master_test_df['source_row'].dropna().astype(int))
    
    common_rows = pred_source_rows & test_source_rows
    only_in_pred = pred_source_rows - test_source_rows
    only_in_test = test_source_rows - pred_source_rows
    
    print(f"Source rows in predictions: {len(pred_source_rows)}")
    print(f"Source rows in master_test: {len(test_source_rows)}")
    print(f"Common source rows: {len(common_rows)}")
    print(f"Only in predictions: {len(only_in_pred)}")
    print(f"Only in master_test: {len(only_in_test)}")
    
    if len(only_in_pred) == 0 and len(only_in_test) == 0:
        print("✓ PASS: All source_rows match between files")
    else:
        print("✗ FAIL: Source rows don't match completely")
        if only_in_pred:
            print(f"  Only in predictions: {list(only_in_pred)[:5]}...")
        if only_in_test:
            print(f"  Only in master_test: {list(only_in_test)[:5]}...")
    
    # =========================================================================
    # VERIFICATION 2: Overall Metrics (from Results.tex lines 42-47)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[3] OVERALL METRICS VERIFICATION")
    print("-" * 40)
    
    # Get true and predicted labels (convert to string first)
    # Ground truth is in 'stance' column, predictions in 'fewshot_label'
    y_true = predictions_df['stance'].astype(str).str.lower().str.strip()
    y_pred = predictions_df['fewshot_label'].astype(str).str.lower().str.strip()
    
    # Normalize labels (handle variants like For, Favour, Nuetral, etc.)
    label_map = {
        'favor': 'favor', 'favour': 'favor', 'for': 'favor',
        'against': 'against',
        'neutral': 'neutral', 'nuetral': 'neutral', 'neautral': 'neutral'
    }
    y_true = y_true.map(lambda x: label_map.get(x, x))
    y_pred = y_pred.map(lambda x: label_map.get(x, x))
    
    # Remove any invalid labels
    valid_mask = y_true.isin(['favor', 'against', 'neutral']) & y_pred.isin(['favor', 'against', 'neutral'])
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    n_samples = len(y_true_valid)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true_valid, y_pred_valid) * 100
    precision_macro = precision_score(y_true_valid, y_pred_valid, average='macro') * 100
    recall_macro = recall_score(y_true_valid, y_pred_valid, average='macro') * 100
    f1_macro = f1_score(y_true_valid, y_pred_valid, average='macro') * 100
    f1_weighted = f1_score(y_true_valid, y_pred_valid, average='weighted') * 100
    
    # Expected values from Results.tex
    expected = {
        'Total test samples': (264, n_samples),
        'Accuracy': (78.0, accuracy),
        'Precision (macro)': (77.6, precision_macro),
        'Recall (macro)': (75.3, recall_macro),
        'F1-Score (macro)': (76.1, f1_macro),
        'F1-Score (weighted)': (77.7, f1_weighted),
    }
    
    print(f"{'Metric':<25} {'Expected':>12} {'Computed':>12} {'Match':>8}")
    print("-" * 60)
    
    all_match = True
    for metric, (exp, comp) in expected.items():
        if metric == 'Total test samples':
            match = exp == comp
            print(f"{metric:<25} {exp:>12} {comp:>12} {'✓' if match else '✗':>8}")
        else:
            match = abs(exp - comp) < 0.5  # Allow 0.5% tolerance for rounding
            print(f"{metric:<25} {exp:>11.1f}% {comp:>11.1f}% {'✓' if match else '✗':>8}")
        if not match:
            all_match = False
    
    if all_match:
        print("\n✓ PASS: All overall metrics match Results.tex")
    else:
        print("\n✗ FAIL: Some overall metrics don't match")
    
    # =========================================================================
    # VERIFICATION 3: Per-Class Metrics (from Results.tex lines 64-66)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[4] PER-CLASS METRICS VERIFICATION")
    print("-" * 40)
    
    # Expected per-class values from Results.tex
    expected_perclass = {
        'against': {'precision': 0.78, 'recall': 0.80, 'f1': 0.79, 'support': 95},
        'favor': {'precision': 0.79, 'recall': 0.86, 'f1': 0.82, 'support': 111},
        'neutral': {'precision': 0.76, 'recall': 0.60, 'f1': 0.67, 'support': 58},
    }
    
    # Compute per-class metrics
    labels = ['against', 'favor', 'neutral']
    precision_perclass = precision_score(y_true_valid, y_pred_valid, labels=labels, average=None)
    recall_perclass = recall_score(y_true_valid, y_pred_valid, labels=labels, average=None)
    f1_perclass = f1_score(y_true_valid, y_pred_valid, labels=labels, average=None)
    
    support_perclass = {}
    for label in labels:
        support_perclass[label] = (y_true_valid == label).sum()
    
    print(f"{'Class':<10} {'Metric':<12} {'Expected':>10} {'Computed':>10} {'Match':>8}")
    print("-" * 55)
    
    perclass_match = True
    for i, label in enumerate(labels):
        exp = expected_perclass[label]
        
        # Precision
        match_p = abs(exp['precision'] - precision_perclass[i]) < 0.015
        print(f"{label:<10} {'Precision':<12} {exp['precision']:>10.2f} {precision_perclass[i]:>10.2f} {'✓' if match_p else '✗':>8}")
        
        # Recall
        match_r = abs(exp['recall'] - recall_perclass[i]) < 0.015
        print(f"{'':<10} {'Recall':<12} {exp['recall']:>10.2f} {recall_perclass[i]:>10.2f} {'✓' if match_r else '✗':>8}")
        
        # F1
        match_f = abs(exp['f1'] - f1_perclass[i]) < 0.015
        print(f"{'':<10} {'F1':<12} {exp['f1']:>10.2f} {f1_perclass[i]:>10.2f} {'✓' if match_f else '✗':>8}")
        
        # Support
        match_s = exp['support'] == support_perclass[label]
        print(f"{'':<10} {'Support':<12} {exp['support']:>10} {support_perclass[label]:>10} {'✓' if match_s else '✗':>8}")
        print()
        
        if not (match_p and match_r and match_f and match_s):
            perclass_match = False
    
    if perclass_match:
        print("✓ PASS: All per-class metrics match Results.tex")
    else:
        print("✗ FAIL: Some per-class metrics don't match")
    
    # =========================================================================
    # VERIFICATION 4: Per-Keyword Accuracy (from Results.tex lines 83-92)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[5] PER-KEYWORD ACCURACY VERIFICATION (Top 10)")
    print("-" * 40)
    
    # Expected per-keyword values from Results.tex
    # Note: keyword names in data use spaces, not underscores
    expected_keywords = {
        'modi': {'accuracy': 94.4, 'f1': 0.91},
        'caa': {'accuracy': 89.5, 'f1': 0.90},
        'congress': {'accuracy': 88.9, 'f1': 0.84},
        'hindutva': {'accuracy': 85.7, 'f1': 0.84},
        'new parliament': {'accuracy': 85.7, 'f1': 0.82},
        'rahulgandhi': {'accuracy': 81.2, 'f1': 0.82},
        'farmers protests': {'accuracy': 80.0, 'f1': 0.82},
        'muslim': {'accuracy': 78.9, 'f1': 0.79},
        'shaheen bagh': {'accuracy': 78.9, 'f1': 0.71},
        'china': {'accuracy': 73.7, 'f1': 0.64},
    }
    
    # Compute per-keyword metrics
    keyword_col = 'keyword'
    if keyword_col not in predictions_df.columns:
        keyword_col = 'matched keyword'
    
    keywords_in_data = predictions_df[keyword_col].str.lower().str.strip().unique()
    print(f"Keywords in data: {sorted(keywords_in_data)}\n")
    
    print(f"{'Keyword':<20} {'Exp Acc':>10} {'Comp Acc':>10} {'Exp F1':>8} {'Comp F1':>8} {'Match':>6}")
    print("-" * 70)
    
    keyword_match = True
    for kw, exp in expected_keywords.items():
        # Get subset for this keyword
        mask = predictions_df[keyword_col].str.lower().str.strip() == kw.lower()
        if mask.sum() == 0:
            print(f"{kw:<20} NOT FOUND IN DATA")
            keyword_match = False
            continue
        
        kw_true = y_true_valid[mask[valid_mask]]
        kw_pred = y_pred_valid[mask[valid_mask]]
        
        if len(kw_true) == 0:
            print(f"{kw:<20} NO VALID SAMPLES")
            keyword_match = False
            continue
        
        kw_accuracy = accuracy_score(kw_true, kw_pred) * 100
        kw_f1 = f1_score(kw_true, kw_pred, average='macro')
        
        match_acc = abs(exp['accuracy'] - kw_accuracy) < 1.0
        match_f1 = abs(exp['f1'] - kw_f1) < 0.02
        
        status = '✓' if (match_acc and match_f1) else '✗'
        print(f"{kw:<20} {exp['accuracy']:>9.1f}% {kw_accuracy:>9.1f}% {exp['f1']:>8.2f} {kw_f1:>8.2f} {status:>6}")
        
        if not (match_acc and match_f1):
            keyword_match = False
    
    if keyword_match:
        print("\n✓ PASS: All per-keyword metrics match Results.tex")
    else:
        print("\n✗ FAIL: Some per-keyword metrics don't match")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=labels)
    print(f"{'':>12} {'against':>10} {'favor':>10} {'neutral':>10}")
    for i, label in enumerate(labels):
        print(f"{label:>12} {cm[i, 0]:>10} {cm[i, 1]:>10} {cm[i, 2]:>10}")
    
    print("\nFull Classification Report:")
    print(classification_report(y_true_valid, y_pred_valid, labels=labels, digits=3))


if __name__ == "__main__":
    main()
