"""
Evaluation and Comparison Script
Compares PyABSA, BERT, and Mistral models with per-keyword breakdown.
Also compares with finetuned Mistral results if available.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
FINETUNED_RESULTS_PATH = BASE_DIR.parent / "4_b_TestingComparison" / "results" / "finetuned_test_results.csv"

# Label order for consistent reporting
LABELS = ["For", "Against", "Neutral"]


def normalize_stance(stance: str) -> str:
    """Normalize stance labels. Returns None for invalid/missing stances."""
    if pd.isna(stance):
        return None
    s = str(stance).lower().strip()
    s = s.replace("favour", "for").replace("favor", "for").replace("nuetral", "neutral")
    if s in ["for"]:
        return "For"
    elif s in ["against"]:
        return "Against"
    elif s in ["neutral"]:
        return "Neutral"
    return None  # Return None for unknown labels


def load_predictions():
    """Load prediction files from all models."""
    predictions = {}
    
    # PyABSA predictions
    pyabsa_path = RESULTS_DIR / "pyabsa_predictions.csv"
    if pyabsa_path.exists():
        predictions['pyabsa'] = pd.read_csv(pyabsa_path)
        predictions['pyabsa']['pred_col'] = 'pyabsa_prediction'
        print(f"✓ Loaded PyABSA predictions: {len(predictions['pyabsa'])} samples")
    
    # BERT predictions
    bert_path = RESULTS_DIR / "bert_predictions.csv"
    if bert_path.exists():
        predictions['bert'] = pd.read_csv(bert_path)
        predictions['bert']['pred_col'] = 'bert_prediction'
        print(f"✓ Loaded BERT predictions: {len(predictions['bert'])} samples")
    
    # Mistral Base predictions
    mistral_base_path = RESULTS_DIR / "mistral_base_predictions.csv"
    if mistral_base_path.exists():
        predictions['mistral_base'] = pd.read_csv(mistral_base_path)
        predictions['mistral_base']['pred_col'] = 'mistral_base_prediction'
        print(f"✓ Loaded Mistral Base predictions: {len(predictions['mistral_base'])} samples")

    # Mistral Few-shot predictions
    mistral_fewshot_path = RESULTS_DIR / "mistral_fewshot_predictions.csv"
    if mistral_fewshot_path.exists():
        predictions['mistral_fewshot'] = pd.read_csv(mistral_fewshot_path)
        predictions['mistral_fewshot']['pred_col'] = 'mistral_prediction'
        print(f"✓ Loaded Mistral Few-shot predictions: {len(predictions['mistral_fewshot'])} samples")
    
    # RoBERTa predictions
    roberta_path = RESULTS_DIR / "roberta_predictions.csv"
    if roberta_path.exists():
        predictions['roberta'] = pd.read_csv(roberta_path)
        predictions['roberta']['pred_col'] = 'roberta_prediction'
        print(f"✓ Loaded RoBERTa predictions: {len(predictions['roberta'])} samples")
    
    # Finetuned Mistral predictions (from 4_b_TestingComparison)
    if FINETUNED_RESULTS_PATH.exists():
        try:
            ft_df = pd.read_csv(FINETUNED_RESULTS_PATH)
            # Normalize column names and extract what we need
            ft_df['original_stance'] = ft_df['stance'].apply(normalize_stance) if 'stance' in ft_df.columns else ft_df['stance_gold'].apply(normalize_stance)
            
            # Get finetuned prediction - use fewshot_label or similar column
            if 'fewshot_label' in ft_df.columns:
                ft_df['mistral_finetuned_prediction'] = ft_df['fewshot_label'].apply(normalize_stance)
            elif 'fewshot_label_for_against' in ft_df.columns:
                ft_df['mistral_finetuned_prediction'] = ft_df['fewshot_label_for_against'].apply(normalize_stance)
            
            if 'mistral_finetuned_prediction' in ft_df.columns:
                # Filter to only the first 265 samples to match baseline models
                # The baseline models were evaluated on 265 test samples
                n_baseline_samples = 265
                if len(ft_df) > n_baseline_samples:
                    print(f"  Note: Filtering finetuned results from {len(ft_df)} to {n_baseline_samples} samples to match baselines")
                    ft_df = ft_df.head(n_baseline_samples)
                # Filter out rows with None/NaN ground truth
                valid_mask = ft_df['original_stance'].notna()
                if (~valid_mask).sum() > 0:
                    print(f"  Note: Skipping {(~valid_mask).sum()} samples with missing ground truth")
                    ft_df = ft_df[valid_mask]
                predictions['mistral_finetuned'] = ft_df
                predictions['mistral_finetuned']['pred_col'] = 'mistral_finetuned_prediction'
                print(f"✓ Loaded Mistral Finetuned predictions: {len(predictions['mistral_finetuned'])} samples")
        except Exception as e:
            print(f"⚠ Could not load finetuned results: {e}")
    
    return predictions


def calculate_metrics(y_true, y_pred, model_name):
    """Calculate comprehensive metrics for a model."""
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', labels=LABELS, zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', labels=LABELS, zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', labels=LABELS, zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', labels=LABELS, zero_division=0),
    }
    
    # Per-class F1 scores
    for label in LABELS:
        y_true_binary = [1 if y == label else 0 for y in y_true]
        y_pred_binary = [1 if y == label else 0 for y in y_pred]
        metrics[f'f1_{label.lower()}'] = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    
    return metrics


def calculate_per_keyword_metrics(predictions_dict):
    """Calculate metrics for each keyword."""
    print("\n" + "=" * 60)
    print("Per-Keyword Performance")
    print("=" * 60)
    
    all_keyword_metrics = []
    
    for model_name, df in predictions_dict.items():
        pred_col = df['pred_col'].iloc[0] if 'pred_col' in df.columns else f'{model_name}_prediction'
        
        if pred_col not in df.columns:
            continue
            
        keywords = df['keyword'].unique()
        
        print(f"\n--- {model_name.upper().replace('_', ' ')} ---")
        print(f"{'Keyword':<20} {'Samples':>8} {'Acc':>8} {'F1':>8}")
        print("-" * 50)
        
        for keyword in sorted(keywords):
            keyword_df = df[df['keyword'] == keyword]
            y_true = keyword_df['original_stance'].tolist()
            y_pred = keyword_df[pred_col].tolist()
            
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro', labels=LABELS, zero_division=0)
            
            print(f"{keyword:<20} {len(keyword_df):>8} {acc:>8.2%} {f1:>8.3f}")
            
            all_keyword_metrics.append({
                'model': model_name,
                'keyword': keyword,
                'samples': len(keyword_df),
                'accuracy': acc,
                'f1_macro': f1
            })
    
    return pd.DataFrame(all_keyword_metrics)


def plot_per_keyword_comparison(keyword_metrics_df):
    """Create per-keyword comparison charts."""
    if keyword_metrics_df.empty:
        return
    
    models = keyword_metrics_df['model'].unique()
    keywords = sorted(keyword_metrics_df['keyword'].unique())
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # Accuracy comparison
    ax1 = axes[0]
    x = np.arange(len(keywords))
    width = 0.8 / len(models)
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
    
    for i, model in enumerate(models):
        model_data = keyword_metrics_df[keyword_metrics_df['model'] == model]
        model_data = model_data.set_index('keyword').reindex(keywords)
        values = model_data['accuracy'].fillna(0).values
        offset = (i - len(models) / 2 + 0.5) * width
        ax1.bar(x + offset, values, width, label=model.upper().replace('_', ' '), color=colors[i % len(colors)])
    
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Per-Keyword Accuracy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(keywords, rotation=45, ha='right')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3)
    
    # F1 comparison
    ax2 = axes[1]
    
    for i, model in enumerate(models):
        model_data = keyword_metrics_df[keyword_metrics_df['model'] == model]
        model_data = model_data.set_index('keyword').reindex(keywords)
        values = model_data['f1_macro'].fillna(0).values
        offset = (i - len(models) / 2 + 0.5) * width
        ax2.bar(x + offset, values, width, label=model.upper().replace('_', ' '), color=colors[i % len(colors)])
    
    ax2.set_ylabel('F1 (Macro)')
    ax2.set_title('Per-Keyword F1 Score Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(keywords, rotation=45, ha='right')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = RESULTS_DIR / "per_keyword_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved per-keyword comparison to: {output_path}")


def plot_confusion_matrices(predictions_dict):
    """Create confusion matrix plots for each model."""
    n_models = len(predictions_dict)
    if n_models == 0:
        return
    
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_models == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    
    for idx, (model_name, df) in enumerate(predictions_dict.items()):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row][col] if n_rows > 1 else axes[col]
        
        pred_col = df['pred_col'].iloc[0] if 'pred_col' in df.columns else f'{model_name}_prediction'
        if pred_col not in df.columns:
            continue
            
        y_true = df['original_stance']
        y_pred = df[pred_col]
        
        cm = confusion_matrix(y_true, y_pred, labels=LABELS)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=LABELS, yticklabels=LABELS, ax=ax)
        ax.set_title(f'{model_name.upper().replace("_", " ")}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                ax.text(j + 0.5, i + 0.7, f'({cm[i, j]})', 
                       ha='center', va='center', fontsize=8, color='gray')
    
    # Hide unused subplots
    for idx in range(n_models, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row][col] if n_rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.tight_layout()
    output_path = RESULTS_DIR / "confusion_matrices.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion matrices to: {output_path}")


def plot_comparison_bar(metrics_list):
    """Create bar chart comparing model performances."""
    if not metrics_list:
        return
    
    df = pd.DataFrame(metrics_list)
    
    plot_metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    metric_names = ['Accuracy', 'F1 (Macro)', 'Precision', 'Recall']
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    x = np.arange(len(plot_metrics))
    width = 0.8 / len(df)
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
    
    for i, (_, row) in enumerate(df.iterrows()):
        values = [row[m] for m in plot_metrics]
        offset = (i - len(df) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, 
                     label=row['model'].upper().replace('_', ' '), 
                     color=colors[i % len(colors)])
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                   f'{val:.2%}', ha='center', va='bottom', fontsize=8, rotation=45)
    
    ax.set_ylabel('Score')
    ax.set_xlabel('Metric')
    ax.set_title('Model Comparison: All Baseline Methods')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.25)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = RESULTS_DIR / "model_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved comparison chart to: {output_path}")


def plot_per_class_f1(metrics_list):
    """Create per-class F1 comparison chart."""
    if not metrics_list:
        return
    
    df = pd.DataFrame(metrics_list)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    x = np.arange(len(LABELS))
    width = 0.8 / len(df)
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
    
    for i, (_, row) in enumerate(df.iterrows()):
        values = [row[f'f1_{label.lower()}'] for label in LABELS]
        offset = (i - len(df) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, 
                     label=row['model'].upper().replace('_', ' '), 
                     color=colors[i % len(colors)])
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                   f'{val:.2%}', ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('F1 Score')
    ax.set_xlabel('Stance Class')
    ax.set_title('Per-Class F1 Score Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.25)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = RESULTS_DIR / "per_class_f1.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved per-class F1 chart to: {output_path}")


def main():
    """Main evaluation pipeline."""
    print("=" * 60)
    print("ABSA Baseline Model Evaluation (All Models)")
    print("=" * 60)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\nLoading predictions...")
    predictions = load_predictions()
    
    if not predictions:
        print("\n✗ No prediction files found!")
        sys.exit(1)
    
    # Calculate metrics for each model
    print("\n" + "=" * 60)
    print("Overall Metrics")
    print("=" * 60)
    
    metrics_list = []
    
    for model_name, df in predictions.items():
        pred_col = df['pred_col'].iloc[0] if 'pred_col' in df.columns else f'{model_name}_prediction'
        
        if pred_col not in df.columns:
            print(f"⚠ Skipping {model_name}: prediction column not found")
            continue
            
        y_true = df['original_stance'].tolist()
        y_pred = df[pred_col].tolist()
        
        metrics = calculate_metrics(y_true, y_pred, model_name)
        metrics_list.append(metrics)
        
        print(f"\n--- {model_name.upper().replace('_', ' ')} ---")
        print(f"Accuracy:        {metrics['accuracy']:.4f}")
        print(f"F1 (Macro):      {metrics['f1_macro']:.4f}")
        print(f"F1 (Weighted):   {metrics['f1_weighted']:.4f}")
        print(f"Precision:       {metrics['precision_macro']:.4f}")
        print(f"Recall:          {metrics['recall_macro']:.4f}")
        print(f"\nPer-class F1:")
        for label in LABELS:
            print(f"  {label}: {metrics[f'f1_{label.lower()}']:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))
    
    # Per-keyword metrics
    keyword_metrics_df = calculate_per_keyword_metrics(predictions)
    
    # Save results
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)
    
    metrics_df = pd.DataFrame(metrics_list)
    metrics_path = RESULTS_DIR / "comparison_report.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✓ Saved overall metrics to: {metrics_path}")
    
    keyword_path = RESULTS_DIR / "per_keyword_metrics.csv"
    keyword_metrics_df.to_csv(keyword_path, index=False)
    print(f"✓ Saved per-keyword metrics to: {keyword_path}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrices(predictions)
    plot_comparison_bar(metrics_list)
    plot_per_class_f1(metrics_list)
    plot_per_keyword_comparison(keyword_metrics_df)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    summary_df = metrics_df[['model', 'accuracy', 'f1_macro', 'f1_weighted']].copy()
    summary_df.columns = ['Model', 'Accuracy', 'F1 (Macro)', 'F1 (Weighted)']
    summary_df = summary_df.sort_values('F1 (Macro)', ascending=False)
    print("\n" + summary_df.to_string(index=False))
    
    # Best model
    best_model = metrics_df.loc[metrics_df['f1_macro'].idxmax(), 'model']
    best_f1 = metrics_df['f1_macro'].max()
    print(f"\n🏆 Best model by F1 (Macro): {best_model.upper().replace('_', ' ')} with {best_f1:.4f}")
    
    # Mistral comparison if both available
    if 'mistral_base' in predictions and 'mistral_finetuned' in predictions:
        base_f1 = metrics_df[metrics_df['model'] == 'mistral_base']['f1_macro'].iloc[0]
        ft_f1 = metrics_df[metrics_df['model'] == 'mistral_finetuned']['f1_macro'].iloc[0]
        improvement = (ft_f1 - base_f1) / base_f1 * 100
        print(f"\n📊 Mistral Finetuning Impact: {improvement:+.1f}% F1 improvement ({base_f1:.3f} → {ft_f1:.3f})")
    
    # Best/worst keywords per model
    print("\n" + "=" * 60)
    print("BEST/WORST KEYWORDS")
    print("=" * 60)
    
    for model in predictions.keys():
        model_kw = keyword_metrics_df[keyword_metrics_df['model'] == model].copy()
        if len(model_kw) > 0:
            best_kw = model_kw.loc[model_kw['f1_macro'].idxmax()]
            worst_kw = model_kw.loc[model_kw['f1_macro'].idxmin()]
            print(f"\n{model.upper().replace('_', ' ')}:")
            print(f"  Best:  {best_kw['keyword']} (F1: {best_kw['f1_macro']:.3f}, n={best_kw['samples']})")
            print(f"  Worst: {worst_kw['keyword']} (F1: {worst_kw['f1_macro']:.3f}, n={worst_kw['samples']})")
    
    print("\n" + "=" * 60)
    print("✓ Evaluation complete!")
    print("=" * 60)
    
    return metrics_df, keyword_metrics_df


if __name__ == "__main__":
    main()
