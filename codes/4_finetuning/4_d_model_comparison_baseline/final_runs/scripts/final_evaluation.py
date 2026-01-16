#!/usr/bin/env python3
"""
Final Evaluation Script for Multi-Model Stance Comparison
Loads all 5 model predictions and generates comprehensive comparison.
Counts unique tweets (not rows) for metrics.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
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

# Set global font sizes for better readability and LaTeX compatibility
plt.rcParams.update({
    'font.size': 18,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'axes.titlesize': 22,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
    'figure.titlesize': 24
})

# Paths
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
LOGS_DIR = SCRIPT_DIR.parent / "logs"

LABELS = ["For", "Against", "Neutral"]
MODEL_CONFIGS = {
    'bert': {'file': 'bert_predictions.csv', 'pred_col': 'bert_prediction'},
    'roberta': {'file': 'roberta_predictions.csv', 'pred_col': 'roberta_prediction'},
    'pyabsa': {'file': 'pyabsa_predictions.csv', 'pred_col': 'pyabsa_prediction'},
    'mistral_base': {'file': 'mistral_base_predictions.csv', 'pred_col': 'mistral_prediction'},
    'mistral_fewshot': {'file': 'mistral_fewshot_predictions.csv', 'pred_col': 'fewshot_label'},
    'mistral_lora': {'file': 'mistral_lora_predictions.csv', 'pred_col': 'fewshot_label'},
}


def setup_logging(log_file: Path):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


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


def load_predictions(results_dir: Path, logger):
    """Load prediction files from all models."""
    predictions = {}
    
    for model_name, config in MODEL_CONFIGS.items():
        pred_path = results_dir / config['file']
        if pred_path.exists():
            df = pd.read_csv(pred_path)
            df['pred_col'] = config['pred_col']
            
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
            logger.info(f"✓ Loaded {model_name}: {len(df)} samples")
        else:
            logger.warning(f"✗ Not found: {pred_path}")
    
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


def calculate_per_keyword_metrics(predictions_dict, logger):
    """Calculate metrics for each keyword."""
    all_keyword_metrics = []
    
    for model_name, df in predictions_dict.items():
        keywords = df['keyword'].unique()
        
        logger.info(f"\n--- {model_name.upper().replace('_', ' ')} ---")
        logger.info(f"{'Keyword':<20} {'Samples':>8} {'Tweets':>8} {'Acc':>8} {'F1':>8}")
        logger.info("-" * 60)
        
        for keyword in sorted(keywords):
            keyword_df = df[df['keyword'] == keyword]
            y_true = keyword_df['original_stance'].tolist()
            y_pred = keyword_df['prediction'].tolist()
            
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro', labels=LABELS, zero_division=0)
            n_tweets = keyword_df['tweet'].nunique()
            
            logger.info(f"{keyword:<20} {len(keyword_df):>8} {n_tweets:>8} {acc:>8.2%} {f1:>8.3f}")
            
            all_keyword_metrics.append({
                'model': model_name,
                'keyword': keyword,
                'samples': len(keyword_df),
                'unique_tweets': n_tweets,
                'accuracy': acc,
                'f1_macro': f1
            })
    
    return pd.DataFrame(all_keyword_metrics)


def plot_confusion_matrices(predictions_dict, output_dir: Path, logger):
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
        
        y_true = df['original_stance']
        y_pred = df['prediction']
        
        cm = confusion_matrix(y_true, y_pred, labels=LABELS)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=LABELS, yticklabels=LABELS, ax=ax)
        # Title removed for LaTeX (caption will be in LaTeX document)
        ax.set_title(f'{model_name.upper().replace("_", " ")}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                ax.text(j + 0.5, i + 0.7, f'({cm[i, j]})', 
                       ha='center', va='center', fontsize=10, color='gray')
    
    # Hide unused subplots
    for idx in range(n_models, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row][col] if n_rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.tight_layout()
    # Save as PDF for LaTeX compatibility
    output_path = output_dir / "confusion_matrices.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    logger.info(f"✓ Saved confusion matrices to: {output_path}")


def plot_mistral_confusion_matrix(predictions_dict, output_dir: Path, logger):
    """Create a standalone high-quality confusion matrix for Mistral LoRA (finetuned model)."""
    if 'mistral_lora' not in predictions_dict:
        logger.warning("Mistral LoRA predictions not found, skipping standalone confusion matrix")
        return
    
    df = predictions_dict['mistral_lora']
    y_true = df['original_stance']
    y_pred = df['prediction']
    
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot heatmap
    sns.heatmap(cm_normalized, annot=False, fmt='.2%', cmap='Blues',
               xticklabels=LABELS, yticklabels=LABELS, ax=ax,
               cbar_kws={'label': 'Proportion'})
    
    # Add annotations with both percentage and count
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            pct = cm_normalized[i, j] * 100
            count = cm[i, j]
            text_color = 'white' if pct > 50 else 'black'
            ax.text(j + 0.5, i + 0.4, f'{pct:.1f}%', 
                   ha='center', va='center', fontsize=14, fontweight='bold', color=text_color)
            ax.text(j + 0.5, i + 0.65, f'(n={count})', 
                   ha='center', va='center', fontsize=10, color=text_color)
    
    # Title removed for LaTeX (caption will be in LaTeX document)
    ax.set_ylabel('True Label', fontsize=14)
    ax.set_xlabel('Predicted Label', fontsize=14)
    
    # Improve tick labels
    ax.set_xticklabels(['Favor', 'Against', 'Neutral'], fontsize=13)
    ax.set_yticklabels(['Favor', 'Against', 'Neutral'], fontsize=13)
    
    plt.tight_layout()
    # Save as PDF for LaTeX compatibility
    output_path = output_dir / "mistral_finetuned_confusion_matrix.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    logger.info(f"✓ Saved Mistral confusion matrix to: {output_path}")


def plot_comparison_bar(metrics_list, output_dir: Path, logger):
    """Create bar chart comparing model performances."""
    if not metrics_list:
        return
    
    df = pd.DataFrame(metrics_list)
    
    plot_metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    metric_names = ['Accuracy', 'F1 (Macro)', 'Precision', 'Recall']
    
    # Large figure for maximum readability
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    
    x = np.arange(len(plot_metrics))
    width = 0.12  # Fixed width for cleaner spacing
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c']
    
    for i, (_, row) in enumerate(df.iterrows()):
        values = [row[m] for m in plot_metrics]
        offset = (i - len(df) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, 
                     label=row['model'].upper().replace('_', ' '), 
                     color=colors[i % len(colors)])
        
        # Large, bold percentage labels above bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                   f'{val:.1%}', ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    # Large font sizes for axis labels
    ax.set_ylabel('Score', fontsize=22, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=22, fontweight='bold')
    # Title removed for LaTeX (caption will be in LaTeX document)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=20, fontweight='bold')
    ax.legend(loc='upper right', fontsize=16, ncol=2)
    ax.set_ylim(0, 1.12)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    # Save as PDF for LaTeX compatibility
    output_path = output_dir / "model_comparison.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    logger.info(f"✓ Saved comparison chart to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Final Multi-Model Evaluation')
    parser.add_argument('--results-dir', type=str, default=str(RESULTS_DIR), help='Results directory')
    parser.add_argument('--log-file', type=str, default=str(LOGS_DIR / 'evaluation.log'), help='Log file')
    args = parser.parse_args()
    
    # Setup
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_path)
    
    results_dir = Path(args.results_dir)
    
    logger.info("=" * 60)
    logger.info("Final Multi-Model Evaluation")
    logger.info("=" * 60)
    
    # Load predictions
    logger.info("\nLoading predictions...")
    predictions = load_predictions(results_dir, logger)
    
    if not predictions:
        logger.error("No prediction files found!")
        sys.exit(1)
    
    # Calculate metrics
    logger.info("\n" + "=" * 60)
    logger.info("Overall Metrics")
    logger.info("=" * 60)
    
    metrics_list = []
    
    for model_name, df in predictions.items():
        y_true = df['original_stance'].tolist()
        y_pred = df['prediction'].tolist()
        
        metrics = calculate_metrics(y_true, y_pred, model_name)
        metrics_list.append(metrics)
        
        n_tweets = df['tweet'].nunique()
        
        logger.info(f"\n--- {model_name.upper().replace('_', ' ')} ---")
        logger.info(f"Samples: {len(df)}, Unique Tweets: {n_tweets}")
        logger.info(f"Accuracy:        {metrics['accuracy']:.4f}")
        logger.info(f"F1 (Macro):      {metrics['f1_macro']:.4f}")
        logger.info(f"F1 (Weighted):   {metrics['f1_weighted']:.4f}")
        logger.info(f"Precision:       {metrics['precision_macro']:.4f}")
        logger.info(f"Recall:          {metrics['recall_macro']:.4f}")
        logger.info(f"\nClassification Report:\n{classification_report(y_true, y_pred, labels=LABELS, zero_division=0)}")
    
    # Per-keyword metrics
    logger.info("\n" + "=" * 60)
    logger.info("Per-Keyword Performance")
    logger.info("=" * 60)
    keyword_metrics_df = calculate_per_keyword_metrics(predictions, logger)
    
    # Save results
    logger.info("\n" + "=" * 60)
    logger.info("Saving Results")
    logger.info("=" * 60)
    
    metrics_df = pd.DataFrame(metrics_list)
    metrics_path = results_dir / "comparison_report.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"✓ Saved overall metrics to: {metrics_path}")
    
    keyword_path = results_dir / "per_keyword_metrics.csv"
    keyword_metrics_df.to_csv(keyword_path, index=False)
    logger.info(f"✓ Saved per-keyword metrics to: {keyword_path}")
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    plot_confusion_matrices(predictions, results_dir, logger)
    plot_comparison_bar(metrics_list, results_dir, logger)
    plot_mistral_confusion_matrix(predictions, results_dir, logger)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    summary_df = metrics_df[['model', 'accuracy', 'f1_macro', 'f1_weighted']].copy()
    summary_df.columns = ['Model', 'Accuracy', 'F1 (Macro)', 'F1 (Weighted)']
    summary_df = summary_df.sort_values('F1 (Macro)', ascending=False)
    logger.info("\n" + summary_df.to_string(index=False))
    
    best_model = metrics_df.loc[metrics_df['f1_macro'].idxmax(), 'model']
    best_f1 = metrics_df['f1_macro'].max()
    logger.info(f"\n🏆 Best model by F1 (Macro): {best_model.upper().replace('_', ' ')} with {best_f1:.4f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ Evaluation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
