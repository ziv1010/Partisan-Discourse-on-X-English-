#!/usr/bin/env python3
"""
Mistral LoRA Per-Keyword Analysis
Generates detailed per-keyword accuracy and F1 scores for Mistral LoRA model only.
"""

import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

# Paths
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"

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


def main():
    # Load Mistral LoRA predictions
    pred_path = RESULTS_DIR / "mistral_lora_predictions.csv"
    
    print("=" * 80)
    print("Mistral LoRA - Per-Keyword Detailed Analysis")
    print("=" * 80)
    
    df = pd.read_csv(pred_path)
    
    # Normalize stances
    df['gt'] = df['stance'].apply(normalize_stance)
    df['pred'] = df['fewshot_label'].apply(normalize_stance)
    
    # Filter valid samples
    valid = df[df['gt'].notna() & df['pred'].notna()]
    
    # Collect results
    results = []
    
    print(f"\n{'Keyword':<20} {'Samples':>8} {'Correct':>8} {'Accuracy':>10} {'F1 Macro':>10}")
    print("-" * 70)
    
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
        
        print(f"{kw:<20} {len(kw_df):>8} {correct:>8} {acc*100:>9.1f}% {f1:>10.3f}")
        
        results.append({
            'keyword': kw,
            'samples': len(kw_df),
            'correct': correct,
            'accuracy_pct': round(acc * 100, 2),
            'f1_macro': round(f1, 4),
            'gt_for': for_gt,
            'gt_against': against_gt,
            'gt_neutral': neutral_gt,
        })
    
    # Totals
    total = len(valid)
    total_correct = sum(r['correct'] for r in results)
    print("-" * 70)
    print(f"{'TOTAL':<20} {total:>8} {total_correct:>8} {total_correct/total*100:>9.1f}%")
    
    # Save to CSV
    results_df = pd.DataFrame(results)
    output_path = RESULTS_DIR / "mistral_lora_per_keyword.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved to: {output_path}")
    
    # Print ground truth distribution
    print("\n" + "=" * 80)
    print("Ground Truth Distribution per Keyword")
    print("=" * 80)
    print(f"\n{'Keyword':<20} {'For':>8} {'Against':>8} {'Neutral':>8} {'Total':>8}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['keyword']:<20} {r['gt_for']:>8} {r['gt_against']:>8} {r['gt_neutral']:>8} {r['samples']:>8}")
    
    print("-" * 70)
    total_for = sum(r['gt_for'] for r in results)
    total_against = sum(r['gt_against'] for r in results)
    total_neutral = sum(r['gt_neutral'] for r in results)
    print(f"{'TOTAL':<20} {total_for:>8} {total_against:>8} {total_neutral:>8} {total:>8}")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
