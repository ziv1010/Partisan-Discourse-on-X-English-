#!/usr/bin/env python3
"""
Evaluate stance classification results.

Computes accuracy, precision, recall, F1-score (macro & per-class),
and generates a confusion matrix visualization.

Usage:
    python evaluate_stance.py --input_csv path/to/results.csv
    python evaluate_stance.py --input_csv path/to/results.csv --ground_truth_col stance --prediction_col fewshot_label
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


def normalize_label(label: str) -> str:
    """Normalize stance labels to canonical form: favor/against/neutral."""
    if pd.isna(label) or label is None:
        return "unknown"
    
    text = str(label).strip().lower()
    
    # Map various spellings to canonical labels
    label_map = {
        # Favor variants
        "for": "favor",
        "favor": "favor",
        "favour": "favor",
        "support": "favor",
        "pro": "favor",
        # Against variants
        "against": "against",
        "oppose": "against",
        "anti": "against",
        # Neutral variants
        "neutral": "neutral",
        "nuetral": "neutral",  # typo in data
        "neither": "neutral",
    }
    
    return label_map.get(text, "unknown")


def evaluate(df: pd.DataFrame, gt_col: str, pred_col: str, output_dir: Path):
    """Compute and save evaluation metrics."""
    
    # Normalize labels
    df["_gt_norm"] = df[gt_col].apply(normalize_label)
    df["_pred_norm"] = df[pred_col].apply(normalize_label)
    
    # Filter out rows with missing ground truth
    valid_mask = df["_gt_norm"].isin(["favor", "against", "neutral"])
    df_valid = df[valid_mask].copy()
    
    if len(df_valid) == 0:
        print("ERROR: No valid ground truth labels found in the dataset.")
        print(f"Ground truth column '{gt_col}' contains: {df[gt_col].unique()[:10]}")
        return
    
    y_true = df_valid["_gt_norm"].values
    y_pred = df_valid["_pred_norm"].values
    
    # Define label order
    labels = ["favor", "against", "neutral"]
    
    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    
    # Handle cases where not all labels are present
    present_labels = sorted(set(y_true) | set(y_pred))
    
    prec_macro = precision_score(y_true, y_pred, labels=present_labels, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, labels=present_labels, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, labels=present_labels, average="macro", zero_division=0)
    
    prec_weighted = precision_score(y_true, y_pred, labels=present_labels, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, labels=present_labels, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=present_labels, average="weighted", zero_division=0)
    
    # Classification report
    report = classification_report(y_true, y_pred, labels=present_labels, zero_division=0, output_dict=True)
    report_str = classification_report(y_true, y_pred, labels=present_labels, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    
    # Summary
    summary = {
        "total_samples": len(df),
        "valid_samples": len(df_valid),
        "skipped_samples": len(df) - len(df_valid),
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_weighted": round(rec_weighted, 4),
        "f1_weighted": round(f1_weighted, 4),
        "per_class": report,
    }
    
    # Print results
    print("=" * 60)
    print("STANCE CLASSIFICATION EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nTotal samples:   {summary['total_samples']}")
    print(f"Valid samples:   {summary['valid_samples']}")
    print(f"Skipped samples: {summary['skipped_samples']} (missing ground truth)")
    print(f"\n{'Metric':<25} {'Value':>10}")
    print("-" * 40)
    print(f"{'Accuracy':<25} {acc:>10.2%}")
    print(f"{'Precision (macro)':<25} {prec_macro:>10.2%}")
    print(f"{'Recall (macro)':<25} {rec_macro:>10.2%}")
    print(f"{'F1-Score (macro)':<25} {f1_macro:>10.2%}")
    print(f"{'F1-Score (weighted)':<25} {f1_weighted:>10.2%}")
    print("\n" + "=" * 60)
    print("PER-CLASS CLASSIFICATION REPORT")
    print("=" * 60)
    print(report_str)
    
    # Save metrics to JSON
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Save classification report to text file
    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write("STANCE CLASSIFICATION EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total samples:   {summary['total_samples']}\n")
        f.write(f"Valid samples:   {summary['valid_samples']}\n")
        f.write(f"Skipped samples: {summary['skipped_samples']}\n\n")
        f.write(f"Accuracy:            {acc:.4f}\n")
        f.write(f"Precision (macro):   {prec_macro:.4f}\n")
        f.write(f"Recall (macro):      {rec_macro:.4f}\n")
        f.write(f"F1-Score (macro):    {f1_macro:.4f}\n")
        f.write(f"F1-Score (weighted): {f1_weighted:.4f}\n\n")
        f.write("=" * 60 + "\n")
        f.write("PER-CLASS CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(report_str)
    print(f"Report saved to: {report_path}")
    
    # Generate confusion matrix plot
    if HAS_PLOTTING:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=present_labels,
            yticklabels=present_labels,
            ax=ax,
        )
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title("Stance Classification Confusion Matrix", fontsize=14)
        plt.tight_layout()
        
        cm_path = output_dir / "confusion_matrix.png"
        fig.savefig(cm_path, dpi=150)
        plt.close(fig)
        print(f"Confusion matrix saved to: {cm_path}")
        
        # Also save normalized confusion matrix
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2%",
            cmap="Blues",
            xticklabels=present_labels,
            yticklabels=present_labels,
            ax=ax2,
        )
        ax2.set_xlabel("Predicted Label", fontsize=12)
        ax2.set_ylabel("True Label", fontsize=12)
        ax2.set_title("Stance Classification Confusion Matrix (Normalized)", fontsize=14)
        plt.tight_layout()
        
        cm_norm_path = output_dir / "confusion_matrix_normalized.png"
        fig2.savefig(cm_norm_path, dpi=150)
        plt.close(fig2)
        print(f"Normalized confusion matrix saved to: {cm_norm_path}")
    else:
        print("\nNote: matplotlib/seaborn not available, skipping plots.")
    
    # Per-keyword analysis
    if "keyword" in df_valid.columns:
        print("\n" + "=" * 60)
        print("PER-KEYWORD ACCURACY")
        print("=" * 60)
        keyword_stats = []
        for kw in df_valid["keyword"].unique():
            kw_mask = df_valid["keyword"] == kw
            kw_true = df_valid.loc[kw_mask, "_gt_norm"]
            kw_pred = df_valid.loc[kw_mask, "_pred_norm"]
            kw_acc = accuracy_score(kw_true, kw_pred)
            kw_f1 = f1_score(kw_true, kw_pred, labels=present_labels, average="macro", zero_division=0)
            keyword_stats.append({
                "keyword": kw,
                "count": len(kw_true),
                "accuracy": kw_acc,
                "f1_macro": kw_f1,
            })
            print(f"  {kw:<25} n={len(kw_true):>4}  acc={kw_acc:.2%}  F1={kw_f1:.2%}")
        
        # Save per-keyword stats
        kw_df = pd.DataFrame(keyword_stats)
        kw_path = output_dir / "per_keyword_metrics.csv"
        kw_df.to_csv(kw_path, index=False)
        print(f"\nPer-keyword metrics saved to: {kw_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate stance classification results")
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to the results CSV file",
    )
    parser.add_argument(
        "--ground_truth_col",
        type=str,
        default="stance",
        help="Column name for ground truth labels (default: stance)",
    )
    parser.add_argument(
        "--prediction_col",
        type=str,
        default="fewshot_label",
        help="Column name for predicted labels (default: fewshot_label)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation outputs (default: same as input)",
    )
    args = parser.parse_args()
    
    input_path = Path(args.input_csv).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = input_path.parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")
    
    # Verify columns exist
    if args.ground_truth_col not in df.columns:
        raise ValueError(f"Ground truth column '{args.ground_truth_col}' not found. Available: {list(df.columns)}")
    if args.prediction_col not in df.columns:
        raise ValueError(f"Prediction column '{args.prediction_col}' not found. Available: {list(df.columns)}")
    
    # Run evaluation
    evaluate(df, args.ground_truth_col, args.prediction_col, output_dir)


if __name__ == "__main__":
    main()
