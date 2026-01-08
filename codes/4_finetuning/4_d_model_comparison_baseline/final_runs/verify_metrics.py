#!/usr/bin/env python3
"""
Script to verify evaluation metrics for Mistral LoRA model from predictions CSV.
Cross-checks the numbers in methodology.tex against actual data.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load predictions
pred_df = pd.read_csv('results/mistral_lora_predictions.csv')

# Check total samples
print(f"Total rows in CSV: {len(pred_df)}")
print(f"Columns: {pred_df.columns.tolist()}")

# Ground truth is in 'stance' column, prediction is in 'fewshot_label'
print(f"\nUnique stance values: {pred_df['stance'].dropna().unique()}")
print(f"Unique fewshot_label values: {pred_df['fewshot_label'].dropna().unique()}")

def normalize_label(label):
    if pd.isna(label):
        return None
    label = str(label).lower().strip()
    if label in ['for', 'favor', 'favour']:
        return 'favor'
    elif label in ['against']:
        return 'against' 
    elif label in ['neutral', 'nuetral']:
        return 'neutral'
    return None

# Check keyword distribution
print(f"\nKeyword distribution:")
print(pred_df['keyword'].value_counts())

# The test set is the first 264-265 samples based on methodology
# Let's filter valid samples and use first 264
test_df = pred_df.head(265).copy()
print(f"\nTest samples (first 265): {len(test_df)}")

# Normalize labels
test_df['gt_normalized'] = test_df['stance'].apply(normalize_label)
test_df['pred_normalized'] = test_df['fewshot_label'].apply(normalize_label)

# Print a few examples to verify
print("\nSample data:")
print(test_df[['keyword', 'stance', 'gt_normalized', 'fewshot_label', 'pred_normalized']].head(10))

# Remove any remaining None values
valid_test = test_df.dropna(subset=['gt_normalized', 'pred_normalized'])
print(f"\nValid test samples after normalization: {len(valid_test)}")

y_true = valid_test['gt_normalized'].tolist()
y_pred = valid_test['pred_normalized'].tolist()

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

print("\n" + "="*60)
print("OVERALL METRICS (for methodology/results tables)")
print("="*60)
print(f"Total test samples: {len(valid_test)}")
print(f"Accuracy: {accuracy*100:.1f}%")
print(f"Precision (macro): {precision_macro*100:.1f}%")
print(f"Recall (macro): {recall_macro*100:.1f}%")
print(f"F1-Score (macro): {f1_macro*100:.1f}%")
print(f"F1-Score (weighted): {f1_weighted*100:.1f}%")

# Per-class metrics
print("\n" + "="*60)
print("PER-CLASS METRICS")
print("="*60)
report = classification_report(y_true, y_pred, digits=2, output_dict=True)
print(classification_report(y_true, y_pred, digits=2))

# Format for LaTeX tables
print("\n" + "="*60)
print("LATEX TABLE FORMAT - Per-Class Classification Performance")
print("="*60)
for cls in ['against', 'favor', 'neutral']:
    r = report[cls]
    print(f"{cls.capitalize()} & {r['precision']:.2f} & {r['recall']:.2f} & {r['f1-score']:.2f} & {int(r['support'])} \\\\")

# Per-keyword metrics
print("\n" + "="*60)
print("PER-KEYWORD ACCURACY (All Keywords)")
print("="*60)

keyword_metrics = []
for kw in valid_test['keyword'].unique():
    kw_mask = valid_test['keyword'] == kw
    kw_true = valid_test.loc[kw_mask, 'gt_normalized'].tolist()
    kw_pred = valid_test.loc[kw_mask, 'pred_normalized'].tolist()
    
    if len(kw_true) > 0:
        kw_acc = accuracy_score(kw_true, kw_pred)
        kw_f1 = f1_score(kw_true, kw_pred, average='macro', zero_division=0)
        keyword_metrics.append({
            'keyword': kw,
            'samples': len(kw_true),
            'accuracy': kw_acc,
            'f1_macro': kw_f1
        })

# Sort by accuracy descending
keyword_metrics.sort(key=lambda x: -x['accuracy'])

print(f"\n{'Keyword':<20} {'Accuracy':>10} {'F1 (macro)':>12} {'Samples':>8}")
print("-" * 52)
for km in keyword_metrics:
    print(f"{km['keyword']:<20} {km['accuracy']*100:>9.1f}% {km['f1_macro']:>11.2f} {km['samples']:>8}")

# Top 10 for LaTeX
print("\n" + "="*60)
print("LATEX TABLE FORMAT - Per-Keyword Classification Accuracy (Top 10)")
print("="*60)
for km in keyword_metrics[:10]:
    kw_escaped = km['keyword'].replace('_', '\\_')
    print(f"{kw_escaped} & {km['accuracy']*100:.1f}\\% & {km['f1_macro']:.2f} \\\\")

# Compare with methodology.tex values
print("\n" + "="*60)
print("COMPARISON WITH METHODOLOGY.TEX VALUES")
print("="*60)
meth_values = {
    'Total test samples': 264,
    'Accuracy': 78.0,
    'Precision (macro)': 78.0,
    'Recall (macro)': 75.3,
    'F1-Score (macro)': 76.2,
    'F1-Score (weighted)': 77.7
}

actual_values = {
    'Total test samples': len(valid_test),
    'Accuracy': accuracy * 100,
    'Precision (macro)': precision_macro * 100,
    'Recall (macro)': recall_macro * 100,
    'F1-Score (macro)': f1_macro * 100,
    'F1-Score (weighted)': f1_weighted * 100
}

print(f"\n{'Metric':<25} {'Methodology':>12} {'Actual':>12} {'Match':>8}")
print("-" * 60)
for metric in meth_values:
    meth_val = meth_values[metric]
    actual_val = actual_values[metric]
    match = "✓" if abs(meth_val - actual_val) < 0.5 else "✗"
    if isinstance(meth_val, int):
        print(f"{metric:<25} {meth_val:>12} {actual_val:>12.0f} {match:>8}")
    else:
        print(f"{metric:<25} {meth_val:>11.1f}% {actual_val:>11.1f}% {match:>8}")

# Per-class comparison with methodology
print("\n" + "="*60)
print("PER-CLASS COMPARISON WITH METHODOLOGY.TEX")
print("="*60)
meth_perclass = {
    'Against': {'precision': 0.78, 'recall': 0.80, 'f1': 0.79, 'support': 95},
    'Favor': {'precision': 0.78, 'recall': 0.86, 'f1': 0.82, 'support': 111},
    'Neutral': {'precision': 0.78, 'recall': 0.60, 'f1': 0.68, 'support': 58}
}

for cls_title, cls_key in [('Against', 'against'), ('Favor', 'favor'), ('Neutral', 'neutral')]:
    meth = meth_perclass[cls_title]
    act = report[cls_key]
    print(f"\n{cls_title}:")
    print(f"  Precision: meth={meth['precision']:.2f} actual={act['precision']:.2f}")
    print(f"  Recall:    meth={meth['recall']:.2f} actual={act['recall']:.2f}")
    print(f"  F1:        meth={meth['f1']:.2f} actual={act['f1-score']:.2f}")
    print(f"  Support:   meth={meth['support']} actual={int(act['support'])}")
