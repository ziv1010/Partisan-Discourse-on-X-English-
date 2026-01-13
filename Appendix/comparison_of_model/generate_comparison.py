#!/usr/bin/env python3
"""
Appendix Model Comparison Script
Generates detailed visualizations and LaTeX tables comparing Mistral LoRA 
finetuned model with all other models, including failure case analysis.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    confusion_matrix,
    classification_report
)

# Paths
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"
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

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10


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


def calculate_metrics(y_true, y_pred):
    """Calculate metrics for a model."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', labels=LABELS, zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', labels=LABELS, zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', labels=LABELS, zero_division=0),
    }


def generate_overall_comparison_table(predictions):
    """Generate LaTeX table comparing all models."""
    metrics_list = []
    
    for model_name, df in predictions.items():
        y_true = df['original_stance'].tolist()
        y_pred = df['prediction'].tolist()
        metrics = calculate_metrics(y_true, y_pred)
        metrics['model'] = MODEL_CONFIGS[model_name]['display']
        metrics['model_key'] = model_name
        metrics_list.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df = metrics_df.sort_values('f1_macro', ascending=False)
    
    # Generate LaTeX table
    latex = r"""% Overall Model Performance Comparison
\begin{table}[htbp]
\centering
\caption{Overall Performance Comparison of Stance Detection Models}
\label{tab:model_comparison}
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{Accuracy} & \textbf{F1 (Macro)} & \textbf{Precision} & \textbf{Recall} \\
\midrule
"""
    
    for _, row in metrics_df.iterrows():
        is_best = row['model_key'] == 'mistral_lora'
        model_name = f"\\textbf{{{row['model']}}}" if is_best else row['model']
        latex += f"{model_name} & {row['accuracy']:.2%} & {row['f1_macro']:.3f} & {row['precision_macro']:.3f} & {row['recall_macro']:.3f} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex, metrics_df


def generate_per_keyword_comparison(predictions):
    """Generate per-keyword comparison between Mistral LoRA and other models."""
    mistral_lora = predictions.get('mistral_lora')
    if mistral_lora is None:
        return "", None
    
    keywords = mistral_lora['keyword'].unique()
    results = []
    
    for keyword in sorted(keywords):
        row = {'keyword': keyword}
        
        for model_name, df in predictions.items():
            keyword_df = df[df['keyword'] == keyword]
            if len(keyword_df) > 0:
                y_true = keyword_df['original_stance'].tolist()
                y_pred = keyword_df['prediction'].tolist()
                acc = accuracy_score(y_true, y_pred)
                row[model_name] = acc
            else:
                row[model_name] = np.nan
        
        results.append(row)
    
    results_df = pd.DataFrame(results)
    
    # Generate LaTeX table
    latex = r"""% Per-Keyword Accuracy Comparison
\begin{table}[htbp]
\centering
\caption{Per-Keyword Accuracy Comparison (\%)}
\label{tab:keyword_comparison}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l""" + "c" * len(MODEL_CONFIGS) + r"""}
\toprule
\textbf{Keyword} """
    
    for model_name in MODEL_CONFIGS.keys():
        latex += f"& \\textbf{{{MODEL_CONFIGS[model_name]['display'].replace('_', ' ')}}} "
    latex += r""" \\
\midrule
"""
    
    for _, row in results_df.iterrows():
        latex += f"{row['keyword'].title()} "
        for model_name in MODEL_CONFIGS.keys():
            val = row.get(model_name, np.nan)
            if not np.isnan(val):
                latex += f"& {val*100:.1f} "
            else:
                latex += "& -- "
        latex += "\\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
}
\end{table}
"""
    return latex, results_df


def find_failure_cases(predictions):
    """Find cases where models fail - especially PyABSA and agreement patterns."""
    mistral_lora = predictions.get('mistral_lora')
    if mistral_lora is None:
        return "", None
    
    # Create a merged dataset with all predictions
    merged = mistral_lora[['tweet', 'keyword', 'original_stance']].copy()
    merged = merged.rename(columns={'original_stance': 'ground_truth'})
    
    for model_name, df in predictions.items():
        pred_df = df[['tweet', 'keyword', 'prediction']].copy()
        pred_df = pred_df.rename(columns={'prediction': f'{model_name}_pred'})
        merged = merged.merge(pred_df, on=['tweet', 'keyword'], how='left')
    
    # Analyze failure patterns
    merged['mistral_lora_correct'] = merged['mistral_lora_pred'] == merged['ground_truth']
    merged['pyabsa_correct'] = merged['pyabsa_pred'] == merged['ground_truth']
    
    # Count agreement patterns
    failure_analysis = {
        'total_samples': len(merged),
        'mistral_lora_correct': merged['mistral_lora_correct'].sum(),
        'pyabsa_correct': merged['pyabsa_correct'].sum(),
        'both_correct': ((merged['mistral_lora_correct']) & (merged['pyabsa_correct'])).sum(),
        'both_wrong': ((~merged['mistral_lora_correct']) & (~merged['pyabsa_correct'])).sum(),
        'lora_only_correct': ((merged['mistral_lora_correct']) & (~merged['pyabsa_correct'])).sum(),
        'pyabsa_only_correct': ((~merged['mistral_lora_correct']) & (merged['pyabsa_correct'])).sum(),
    }
    
    # Find specific failure examples (where all models fail)
    all_models = list(predictions.keys())
    merged['all_wrong'] = True
    for model_name in all_models:
        col = f'{model_name}_pred'
        if col in merged.columns:
            merged['all_wrong'] = merged['all_wrong'] & (merged[col] != merged['ground_truth'])
    
    all_fail_cases = merged[merged['all_wrong']][['tweet', 'keyword', 'ground_truth'] + 
                                                   [f'{m}_pred' for m in all_models if f'{m}_pred' in merged.columns]]
    
    # Find cases where PyABSA fails but Mistral LoRA succeeds
    pyabsa_fail_lora_success = merged[
        (merged['mistral_lora_correct']) & (~merged['pyabsa_correct'])
    ][['tweet', 'keyword', 'ground_truth', 'mistral_lora_pred', 'pyabsa_pred']]
    
    return merged, failure_analysis, all_fail_cases, pyabsa_fail_lora_success


def generate_failure_analysis_table(failure_analysis):
    """Generate LaTeX table for failure analysis."""
    latex = r"""% Model Agreement Analysis
\begin{table}[htbp]
\centering
\caption{Agreement Analysis: Mistral LoRA vs PyABSA}
\label{tab:agreement_analysis}
\begin{tabular}{lcc}
\toprule
\textbf{Category} & \textbf{Count} & \textbf{Percentage} \\
\midrule
"""
    total = failure_analysis['total_samples']
    
    rows = [
        ('Both Correct', failure_analysis['both_correct']),
        ('Both Wrong', failure_analysis['both_wrong']),
        ('Mistral LoRA Only Correct', failure_analysis['lora_only_correct']),
        ('PyABSA Only Correct', failure_analysis['pyabsa_only_correct']),
    ]
    
    for name, count in rows:
        pct = count / total * 100
        latex += f"{name} & {count} & {pct:.1f}\\% \\\\\n"
    
    latex += f"\\midrule\n\\textbf{{Total Samples}} & {total} & 100.0\\% \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_failure_examples_table(all_fail_cases, max_examples=10):
    """Generate LaTeX table with examples where all models fail."""
    latex = r"""% Examples Where All Models Fail
\begin{table}[htbp]
\centering
\caption{Sample Cases Where All Models Fail (Truncated)}
\label{tab:failure_examples}
\resizebox{\textwidth}{!}{%
\begin{tabular}{p{6cm}lcc}
\toprule
\textbf{Tweet (Truncated)} & \textbf{Keyword} & \textbf{Ground Truth} & \textbf{Common Prediction} \\
\midrule
"""
    
    for idx, row in all_fail_cases.head(max_examples).iterrows():
        tweet = row['tweet'][:80].replace('&', '\\&').replace('%', '\\%').replace('#', '\\#').replace('_', '\\_') + "..."
        keyword = row['keyword'].title()
        gt = row['ground_truth']
        # Find most common wrong prediction
        preds = [row[col] for col in all_fail_cases.columns if col.endswith('_pred') and pd.notna(row[col])]
        common_pred = max(set(preds), key=preds.count) if preds else 'N/A'
        
        latex += f"{tweet} & {keyword} & {gt} & {common_pred} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
}
\end{table}
"""
    return latex


def plot_model_comparison_heatmap(predictions, output_dir):
    """Create heatmap showing per-keyword accuracy for all models."""
    keywords = sorted(predictions['mistral_lora']['keyword'].unique())
    models = list(MODEL_CONFIGS.keys())
    
    # Create accuracy matrix
    accuracy_matrix = np.zeros((len(keywords), len(models)))
    
    for j, model_name in enumerate(models):
        df = predictions.get(model_name)
        if df is not None:
            for i, keyword in enumerate(keywords):
                keyword_df = df[df['keyword'] == keyword]
                if len(keyword_df) > 0:
                    y_true = keyword_df['original_stance'].tolist()
                    y_pred = keyword_df['prediction'].tolist()
                    accuracy_matrix[i, j] = accuracy_score(y_true, y_pred)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 10))
    
    sns.heatmap(accuracy_matrix * 100, 
                annot=True, 
                fmt='.0f',
                cmap='RdYlGn',
                xticklabels=[MODEL_CONFIGS[m]['display'] for m in models],
                yticklabels=[k.title() for k in keywords],
                ax=ax,
                vmin=0, vmax=100,
                cbar_kws={'label': 'Accuracy (%)'})
    
    ax.set_title('Per-Keyword Accuracy Comparison Across Models', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Keyword', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_path = output_dir / "keyword_accuracy_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_mistral_vs_others(predictions, output_dir):
    """Create scatter plot comparing Mistral LoRA with each other model."""
    mistral_lora = predictions.get('mistral_lora')
    if mistral_lora is None:
        return
    
    keywords = sorted(mistral_lora['keyword'].unique())
    other_models = [m for m in MODEL_CONFIGS.keys() if m != 'mistral_lora']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Calculate Mistral LoRA accuracy per keyword
    lora_acc = {}
    for keyword in keywords:
        keyword_df = mistral_lora[mistral_lora['keyword'] == keyword]
        if len(keyword_df) > 0:
            y_true = keyword_df['original_stance'].tolist()
            y_pred = keyword_df['prediction'].tolist()
            lora_acc[keyword] = accuracy_score(y_true, y_pred)
    
    for idx, model_name in enumerate(other_models):
        ax = axes[idx]
        df = predictions.get(model_name)
        
        if df is not None:
            other_acc = {}
            for keyword in keywords:
                keyword_df = df[df['keyword'] == keyword]
                if len(keyword_df) > 0:
                    y_true = keyword_df['original_stance'].tolist()
                    y_pred = keyword_df['prediction'].tolist()
                    other_acc[keyword] = accuracy_score(y_true, y_pred)
            
            # Plot scatter
            x = [lora_acc.get(k, 0) * 100 for k in keywords]
            y = [other_acc.get(k, 0) * 100 for k in keywords]
            
            ax.scatter(x, y, alpha=0.7, s=100, c='steelblue', edgecolors='navy')
            
            # Add diagonal line
            ax.plot([0, 100], [0, 100], 'r--', alpha=0.5, label='Equal Performance')
            
            # Add keyword labels
            for i, keyword in enumerate(keywords):
                ax.annotate(keyword[:8], (x[i], y[i]), fontsize=7, alpha=0.7)
            
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_xlabel('Mistral LoRA Accuracy (%)')
            ax.set_ylabel(f'{MODEL_CONFIGS[model_name]["display"]} Accuracy (%)')
            ax.set_title(f'vs {MODEL_CONFIGS[model_name]["display"]}')
            ax.grid(True, alpha=0.3)
    
    # Hide empty subplot if odd number of models
    if len(other_models) < len(axes):
        for idx in range(len(other_models), len(axes)):
            axes[idx].set_visible(False)
    
    plt.suptitle('Mistral LoRA vs Other Models (Per-Keyword Accuracy)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / "mistral_vs_others_scatter.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_per_class_f1(predictions, output_dir):
    """Create grouped bar chart showing F1 score per class for each model."""
    models = list(MODEL_CONFIGS.keys())
    
    f1_scores = {label: [] for label in LABELS}
    
    for model_name in models:
        df = predictions.get(model_name)
        if df is not None:
            y_true = df['original_stance'].tolist()
            y_pred = df['prediction'].tolist()
            
            for label in LABELS:
                y_true_binary = [1 if y == label else 0 for y in y_true]
                y_pred_binary = [1 if y == label else 0 for y in y_pred]
                f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
                f1_scores[label].append(f1)
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.25
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    
    for i, label in enumerate(LABELS):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, f1_scores[label], width, label=label, color=colors[i])
        
        for bar, val in zip(bars, f1_scores[label]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('F1 Score')
    ax.set_xlabel('Model')
    ax.set_title('Per-Class F1 Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_CONFIGS[m]['display'] for m in models], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / "per_class_f1_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_confusion_matrix_comparison(predictions, output_dir):
    """Create side-by-side confusion matrices for best and worst models."""
    # Find best and worst
    metrics_list = []
    for model_name, df in predictions.items():
        y_true = df['original_stance'].tolist()
        y_pred = df['prediction'].tolist()
        f1 = f1_score(y_true, y_pred, average='macro', labels=LABELS, zero_division=0)
        metrics_list.append({'model': model_name, 'f1': f1})
    
    metrics_df = pd.DataFrame(metrics_list).sort_values('f1', ascending=False)
    best_model = metrics_df.iloc[0]['model']
    worst_model = metrics_df.iloc[-1]['model']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (model_name, ax, title_suffix) in enumerate([
        (best_model, axes[0], '(Best)'),
        (worst_model, axes[1], '(Worst)')
    ]):
        df = predictions[model_name]
        y_true = df['original_stance']
        y_pred = df['prediction']
        
        cm = confusion_matrix(y_true, y_pred, labels=LABELS)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='Blues',
                   xticklabels=LABELS, yticklabels=LABELS, ax=ax)
        
        # Add raw counts
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                ax.text(j + 0.5, i + 0.72, f'({cm[i, j]})', 
                       ha='center', va='center', fontsize=8, color='gray')
        
        ax.set_title(f'{MODEL_CONFIGS[model_name]["display"]} {title_suffix}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    plt.suptitle('Confusion Matrix: Best vs Worst Model', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / "confusion_matrix_best_vs_worst.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def main():
    print("=" * 60)
    print("Appendix Model Comparison Generator")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load predictions
    print("\nLoading predictions...")
    predictions = load_all_predictions()
    
    if not predictions:
        print("ERROR: No prediction files found!")
        sys.exit(1)
    
    # Generate LaTeX tables
    print("\nGenerating LaTeX tables...")
    
    latex_content = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{float}

\begin{document}

\section*{Appendix: Model Comparison}

This appendix provides a detailed comparison of stance detection models, 
with a focus on comparing the best-performing model (Mistral LoRA fine-tuned) 
with other approaches including PyABSA.

"""
    
    # Overall comparison table
    overall_latex, overall_df = generate_overall_comparison_table(predictions)
    latex_content += overall_latex + "\n\n"
    
    # Per-keyword comparison table
    keyword_latex, keyword_df = generate_per_keyword_comparison(predictions)
    latex_content += keyword_latex + "\n\n"
    
    # Failure analysis
    merged, failure_analysis, all_fail_cases, pyabsa_fail_cases = find_failure_cases(predictions)
    
    failure_latex = generate_failure_analysis_table(failure_analysis)
    latex_content += failure_latex + "\n\n"
    
    if len(all_fail_cases) > 0:
        examples_latex = generate_failure_examples_table(all_fail_cases)
        latex_content += examples_latex + "\n\n"
    
    # Add figure references
    latex_content += r"""
\subsection*{Visualizations}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{keyword_accuracy_heatmap.png}
\caption{Per-keyword accuracy heatmap across all models.}
\label{fig:heatmap}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{mistral_vs_others_scatter.png}
\caption{Scatter plots comparing Mistral LoRA accuracy with each other model.}
\label{fig:scatter}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{per_class_f1_comparison.png}
\caption{Per-class F1 score comparison across models.}
\label{fig:f1}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{confusion_matrix_best_vs_worst.png}
\caption{Confusion matrices for best (Mistral LoRA) and worst performing models.}
\label{fig:cm}
\end{figure}

\end{document}
"""
    
    # Save LaTeX file
    latex_path = OUTPUT_DIR / "appendix_tables.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_content)
    print(f"✓ Saved LaTeX: {latex_path}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    plot_model_comparison_heatmap(predictions, OUTPUT_DIR)
    plot_mistral_vs_others(predictions, OUTPUT_DIR)
    plot_per_class_f1(predictions, OUTPUT_DIR)
    plot_confusion_matrix_comparison(predictions, OUTPUT_DIR)
    
    # Save failure cases to CSV for manual review
    if len(all_fail_cases) > 0:
        fail_path = OUTPUT_DIR / "all_models_fail_cases.csv"
        all_fail_cases.to_csv(fail_path, index=False)
        print(f"✓ Saved: {fail_path}")
    
    if len(pyabsa_fail_cases) > 0:
        pyabsa_path = OUTPUT_DIR / "pyabsa_fail_lora_success.csv"
        pyabsa_fail_cases.to_csv(pyabsa_path, index=False)
        print(f"✓ Saved: {pyabsa_path}")
    
    # Save per-keyword comparison
    if keyword_df is not None:
        keyword_path = OUTPUT_DIR / "keyword_accuracy_matrix.csv"
        keyword_df.to_csv(keyword_path, index=False)
        print(f"✓ Saved: {keyword_path}")
    
    print("\n" + "=" * 60)
    print("✓ All outputs generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
