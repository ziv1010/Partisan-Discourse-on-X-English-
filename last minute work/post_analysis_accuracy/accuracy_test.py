#!/usr/bin/env python3
"""
Post-Analysis Accuracy Test

Compares model predictions (fewshot_label_for_against) with human annotations (STANCE)
from the multi-sheet Excel file.

Outputs:
- Per-keyword accuracy report
- Overall accuracy statistics
- Classification report
- Confusion matrix
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


def normalize_stance(stance):
    """Normalize stance labels to lowercase standard format."""
    if pd.isna(stance):
        return None
    s = str(stance).lower().strip()
    if s in ['for', 'favor', 'favour']:
        return 'favor'
    elif s in ['against']:
        return 'against'
    elif s in ['neutral', 'nuetral']:
        return 'neutral'
    return None


def main():
    # Paths
    script_dir = Path(__file__).parent
    input_file = script_dir / "Annotations_Jan'26-2.xlsx"
    
    print("=" * 70)
    print("POST-ANALYSIS ACCURACY TEST")
    print("=" * 70)
    print(f"\nInput file: {input_file}")
    
    # Load all sheets
    xl = pd.ExcelFile(input_file)
    
    # Skip the Legend sheet
    data_sheets = [s for s in xl.sheet_names if s != 'Legend']
    
    all_data = []
    
    print(f"\nLoading {len(data_sheets)} keyword sheets...")
    
    for sheet in data_sheets:
        df = pd.read_excel(xl, sheet_name=sheet)
        
        # Find the STANCE column (human annotation)
        stance_col = None
        for col in df.columns:
            if 'STANCE' in col.upper() and col.upper() != 'STANCE_GOLD':
                stance_col = col
                break
        
        if stance_col is None or 'fewshot_label_for_against' not in df.columns:
            print(f"  Skipping {sheet}: missing required columns")
            continue
        
        # Get keyword from sheet name or column
        keyword = sheet.strip()
        
        # Extract relevant columns
        subset = df[['fewshot_label_for_against', stance_col]].copy()
        subset.columns = ['model_prediction', 'human_annotation']
        subset['keyword'] = keyword
        
        # Normalize stances
        subset['model_prediction'] = subset['model_prediction'].apply(normalize_stance)
        subset['human_annotation'] = subset['human_annotation'].apply(normalize_stance)
        
        # Filter out invalid rows
        valid_mask = subset['model_prediction'].notna() & subset['human_annotation'].notna()
        subset = subset[valid_mask]
        
        print(f"  {sheet}: {len(subset)} valid samples")
        all_data.append(subset)
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal valid samples: {len(combined_df)}")
    
    # =========================================================================
    # OVERALL METRICS
    # =========================================================================
    print("\n" + "=" * 70)
    print("OVERALL METRICS")
    print("=" * 70)
    
    y_true = combined_df['human_annotation']
    y_pred = combined_df['model_prediction']
    labels = ['favor', 'against', 'neutral']
    
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0)
    
    print(f"\nTotal samples:       {len(combined_df)}")
    print(f"Accuracy:            {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"Precision (macro):   {precision_macro:.4f} ({precision_macro*100:.1f}%)")
    print(f"Recall (macro):      {recall_macro:.4f} ({recall_macro*100:.1f}%)")
    print(f"F1-Score (macro):    {f1_macro:.4f} ({f1_macro*100:.1f}%)")
    print(f"F1-Score (weighted): {f1_weighted:.4f} ({f1_weighted*100:.1f}%)")
    
    # Per-class metrics
    print("\n" + "-" * 40)
    print("Per-Class Metrics:")
    print("-" * 40)
    print(classification_report(y_true, y_pred, labels=labels, digits=3, zero_division=0))
    
    # =========================================================================
    # PER-KEYWORD METRICS
    # =========================================================================
    print("\n" + "=" * 70)
    print("PER-KEYWORD ACCURACY")
    print("=" * 70)
    
    keyword_metrics = []
    
    print(f"\n{'Keyword':<25} {'Samples':>8} {'Accuracy':>10} {'F1 (macro)':>12}")
    print("-" * 60)
    
    for keyword in sorted(combined_df['keyword'].unique()):
        kw_mask = combined_df['keyword'] == keyword
        kw_true = y_true[kw_mask]
        kw_pred = y_pred[kw_mask]
        
        kw_acc = accuracy_score(kw_true, kw_pred)
        kw_f1 = f1_score(kw_true, kw_pred, average='macro', labels=labels, zero_division=0)
        
        print(f"{keyword:<25} {len(kw_true):>8} {kw_acc:>9.1%} {kw_f1:>12.3f}")
        
        keyword_metrics.append({
            'keyword': keyword,
            'samples': len(kw_true),
            'accuracy': kw_acc,
            'f1_macro': kw_f1
        })
    
    # =========================================================================
    # CONFUSION MATRIX
    # =========================================================================
    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(f"\n{'':>12} {'favor':>10} {'against':>10} {'neutral':>10}")
    for i, label in enumerate(labels):
        print(f"{label:>12} {cm[i, 0]:>10} {cm[i, 1]:>10} {cm[i, 2]:>10}")
    
    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)
    
    # Save overall metrics
    overall_metrics = {
        'total_samples': len(combined_df),
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }
    overall_df = pd.DataFrame([overall_metrics])
    overall_path = script_dir / "overall_accuracy_report.csv"
    overall_df.to_csv(overall_path, index=False)
    print(f"✓ Saved overall metrics to: {overall_path}")
    
    # Save per-keyword metrics
    keyword_df = pd.DataFrame(keyword_metrics)
    keyword_path = script_dir / "per_keyword_accuracy_report.csv"
    keyword_df.to_csv(keyword_path, index=False)
    print(f"✓ Saved per-keyword metrics to: {keyword_path}")
    
    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    sns.heatmap(cm_normalized, annot=False, cmap='Blues',
               xticklabels=['Favor', 'Against', 'Neutral'],
               yticklabels=['Favor', 'Against', 'Neutral'], ax=ax)
    
    # Add annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            pct = cm_normalized[i, j] * 100
            count = cm[i, j]
            text_color = 'white' if pct > 50 else 'black'
            ax.text(j + 0.5, i + 0.4, f'{pct:.1f}%', 
                   ha='center', va='center', fontsize=14, fontweight='bold', color=text_color)
            ax.text(j + 0.5, i + 0.65, f'(n={count})', 
                   ha='center', va='center', fontsize=10, color=text_color)
    
    ax.set_title('Post-Analysis Accuracy: Model vs Human Annotation', fontsize=14, fontweight='bold')
    ax.set_ylabel('Human Annotation', fontsize=12)
    ax.set_xlabel('Model Prediction', fontsize=12)
    
    plt.tight_layout()
    cm_path = script_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion matrix to: {cm_path}")
    
    # Save detailed report
    report_path = script_dir / "detailed_accuracy_report.txt"
    with open(report_path, 'w') as f:
        f.write("POST-ANALYSIS ACCURACY REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Input file: {input_file}\n")
        f.write(f"Total samples: {len(combined_df)}\n\n")
        
        f.write("OVERALL METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy:            {accuracy:.4f} ({accuracy*100:.1f}%)\n")
        f.write(f"Precision (macro):   {precision_macro:.4f} ({precision_macro*100:.1f}%)\n")
        f.write(f"Recall (macro):      {recall_macro:.4f} ({recall_macro*100:.1f}%)\n")
        f.write(f"F1-Score (macro):    {f1_macro:.4f} ({f1_macro*100:.1f}%)\n")
        f.write(f"F1-Score (weighted): {f1_weighted:.4f} ({f1_weighted*100:.1f}%)\n\n")
        
        f.write("CLASSIFICATION REPORT\n")
        f.write("-" * 40 + "\n")
        f.write(classification_report(y_true, y_pred, labels=labels, digits=3, zero_division=0))
        f.write("\n\n")
        
        f.write("PER-KEYWORD ACCURACY\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Keyword':<25} {'Samples':>8} {'Accuracy':>10} {'F1':>8}\n")
        for m in keyword_metrics:
            f.write(f"{m['keyword']:<25} {m['samples']:>8} {m['accuracy']:>9.1%} {m['f1_macro']:>8.3f}\n")
    
    print(f"✓ Saved detailed report to: {report_path}")
    
    print("\n" + "=" * 70)
    print("✓ Accuracy test complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
