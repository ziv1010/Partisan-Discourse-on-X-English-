#!/usr/bin/env python3
"""McNemar significance tests for Hindi stance evaluation: Mistral LoRA vs GPT vs Gemini."""

from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import openpyxl


def normalize_stance(value: str) -> Optional[str]:
    v = value.strip().lower().replace("favour", "favor")
    mapping = {
        "favor": "favor", "for": "favor", "in favor": "favor",
        "against": "against", "negative": "against",
        "neutral": "neutral", "nuetral": "neutral",
    }
    return mapping.get(v)


def load_gold(path: Path) -> Dict[int, str]:
    gold = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            label = normalize_stance(row["annotator_1_stance"])
            if label:
                gold[int(row["tweet_id"])] = label
    return gold


def load_sno_csv(path: Path) -> Dict[int, str]:
    preds = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = list(row.keys())[0]
            val = list(row.values())[1]
            label = normalize_stance(val)
            if label:
                preds[int(row[key].strip())] = label
    return preds


def load_sno_xlsx(path: Path) -> Dict[int, str]:
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb.active
    preds = {}
    header = None
    for r in ws.iter_rows(values_only=True):
        if header is None:
            header = r
            continue
        label = normalize_stance(str(r[1]))
        if label:
            preds[int(r[0])] = label
    wb.close()
    return preds


def load_mistral_hindi_seed(path: Path) -> Dict[int, str]:
    preds = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            label = normalize_stance(row["fewshot_label"])
            if label:
                preds[int(row["tweet_id"])] = label
    return preds


def mcnemar_exact(a_right_b_wrong: int, a_wrong_b_right: int) -> float:
    n = a_right_b_wrong + a_wrong_b_right
    if n == 0:
        return 1.0
    k = min(a_right_b_wrong, a_wrong_b_right)
    p = 0.0
    for i in range(k + 1):
        p += math.comb(n, i) * (0.5 ** n)
    return min(2.0 * p, 1.0)


def holm_correction(p_values: List[float]) -> List[float]:
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [0.0] * m
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted = p * (m - rank)
        adjusted = min(adjusted, 1.0)
        cummax = max(cummax, adjusted)
        corrected[orig_idx] = cummax
    return corrected


def pairwise_mcnemar(
    model_preds: Dict[str, Dict[int, str]],
    gold: Dict[int, str],
) -> List[dict]:
    models = sorted(model_preds.keys())
    rows = []
    for a, b in combinations(models, 2):
        preds_a, preds_b = model_preds[a], model_preds[b]
        common = sorted(set(preds_a) & set(preds_b) & set(gold))
        a_right_b_wrong = 0
        a_wrong_b_right = 0
        for idx in common:
            g = gold[idx]
            ca = preds_a[idx] == g
            cb = preds_b[idx] == g
            if ca and not cb:
                a_right_b_wrong += 1
            elif cb and not ca:
                a_wrong_b_right += 1

        acc_a = sum(1 for i in common if preds_a[i] == gold[i]) / len(common)
        acc_b = sum(1 for i in common if preds_b[i] == gold[i]) / len(common)
        p = mcnemar_exact(a_right_b_wrong, a_wrong_b_right)
        better = a if acc_a >= acc_b else b

        rows.append({
            "model_a": a,
            "model_b": b,
            "n_compared": len(common),
            "a_correct_b_wrong": a_right_b_wrong,
            "a_wrong_b_correct": a_wrong_b_right,
            "acc_a": round(acc_a, 4),
            "acc_b": round(acc_b, 4),
            "accuracy_diff": round(acc_a - acc_b, 4),
            "better_model": better,
            "exact_p": p,
        })
    return rows


def main():
    base = Path(__file__).resolve().parent
    chatgpt_gemini = base.parent / "CHATGPT+GEMINI"
    gold_path = base / "hindi_stance_annotation_100.csv"

    gold = load_gold(gold_path)
    print(f"Gold labels loaded: {len(gold)}")

    model_preds: Dict[str, Dict[int, str]] = {}

    # Mistral LoRA english eval (Hindi tweets translated to English, as reported in paper)
    for seed in [13, 42, 100]:
        p = base / f"mistral_lora_multiseed_english_eval/seed_{seed}/mistral_lora_predictions.csv"
        model_preds[f"mistral_lora_hi_seed{seed}"] = load_mistral_hindi_seed(p)

    # ChatGPT fewshot: run1 = chatgpt_hindi_stance, run2, run3
    model_preds["chatgpt_fewshot_hi_run1"] = load_sno_csv(chatgpt_gemini / "chatgpt_hindi_stance.csv")
    model_preds["chatgpt_fewshot_hi_run2"] = load_sno_xlsx(chatgpt_gemini / "chatgpt_fewshot_hi_run2.xlsx")
    model_preds["chatgpt_fewshot_hi_run3"] = load_sno_xlsx(chatgpt_gemini / "chatgpt_fewshot_hi_run3.xlsx")

    # ChatGPT zeroshot: run1 = chatgpt_hindi_zeroshot_stance_results, run2, run3
    model_preds["chatgpt_zeroshot_hi_run1"] = load_sno_xlsx(chatgpt_gemini / "chatgpt_hindi_zeroshot_stance_results.xlsx")
    model_preds["chatgpt_zeroshot_hi_run2"] = load_sno_csv(chatgpt_gemini / "chatgpt_zeroshot_hi_run2.csv")
    model_preds["chatgpt_zeroshot_hi_run3"] = load_sno_csv(chatgpt_gemini / "chatgpt_zeroshot_hi_run3.csv")

    # Gemini fewshot: run1 = gemini_hindi_stance, run2, run3
    model_preds["gemini_fewshot_hi_run1"] = load_sno_csv(chatgpt_gemini / "gemini_hindi_stance.csv")
    model_preds["gemini_fewshot_hi_run2"] = load_sno_xlsx(chatgpt_gemini / "gemini_fewshot_hi_run2.xlsx")
    model_preds["gemini_fewshot_hi_run3"] = load_sno_xlsx(chatgpt_gemini / "gemini_fewshot_hi_run3.xlsx")

    # Gemini zeroshot: run1 = gemini_hindi_zeroshot_stance_results, run2, run3 (using CSV for run3)
    model_preds["gemini_zeroshot_hi_run1"] = load_sno_xlsx(chatgpt_gemini / "gemini_hindi_zeroshot_stance_results.xlsx")
    model_preds["gemini_zeroshot_hi_run2"] = load_sno_xlsx(chatgpt_gemini / "gemini_zeroshot_hi_run2.xlsx")
    model_preds["gemini_zeroshot_hi_run3"] = load_sno_csv(chatgpt_gemini / "gemini_zeroshot_hi_run3.csv")

    # Print accuracy summary
    print(f"\n{'Model':<35s} {'Acc':>6s}  {'n':>3s}")
    print("-" * 48)
    for name in sorted(model_preds):
        preds = model_preds[name]
        common = set(preds) & set(gold)
        acc = sum(1 for i in common if preds[i] == gold[i]) / len(common)
        print(f"{name:<35s} {acc:6.2%}  {len(common):3d}")

    # Pairwise McNemar
    rows = pairwise_mcnemar(model_preds, gold)

    # Holm correction
    p_values = [r["exact_p"] for r in rows]
    holm_p = holm_correction(p_values)
    for r, hp in zip(rows, holm_p):
        r["holm_p"] = hp
        r["holm_significant_0.05"] = hp < 0.05

    # Write CSV
    out_csv = base / "mcnemar_tests_hindi.csv"
    fields = ["model_a", "model_b", "n_compared", "a_correct_b_wrong", "a_wrong_b_correct",
              "acc_a", "acc_b", "accuracy_diff", "better_model", "exact_p", "holm_p", "holm_significant_0.05"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out_csv}")

    # Write Markdown
    out_md = base / "mcnemar_tests_hindi.md"
    with open(out_md, "w") as f:
        f.write("# McNemar Tests: Hindi Stance Evaluation\n\n")
        f.write("Pairwise McNemar exact tests with Holm correction.\n")
        f.write(f"Gold labels: `annotator_1_stance` from `hindi_stance_annotation_100.csv` (n={len(gold)})\n\n")
        f.write("**Note:** `gemini_zeroshot_hi_run3` uses the CSV version (16 predictions differ from the XLSX).\n\n")

        # Summary table
        f.write("## Accuracy Summary\n\n")
        f.write("| Model | Accuracy | n |\n| --- | --- | --- |\n")
        for name in sorted(model_preds):
            preds = model_preds[name]
            common = set(preds) & set(gold)
            acc = sum(1 for i in common if preds[i] == gold[i]) / len(common)
            f.write(f"| {name} | {acc:.2%} | {len(common)} |\n")

        f.write("\n## Pairwise McNemar Tests\n\n")
        f.write("| model_a | model_b | n | a\\_right\\_b\\_wrong | a\\_wrong\\_b\\_right | acc\\_a | acc\\_b | diff | better | exact\\_p | holm\\_p | sig |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for r in rows:
            sig = "**Yes**" if r["holm_significant_0.05"] else "No"
            f.write(f"| {r['model_a']} | {r['model_b']} | {r['n_compared']} | {r['a_correct_b_wrong']} | {r['a_wrong_b_correct']} | {r['acc_a']:.4f} | {r['acc_b']:.4f} | {r['accuracy_diff']:+.4f} | {r['better_model']} | {r['exact_p']:.4e} | {r['holm_p']:.4e} | {sig} |\n")

    print(f"Wrote {out_md}")

    # Print significant results
    sig_rows = [r for r in rows if r["holm_significant_0.05"]]
    print(f"\n=== {len(sig_rows)} significant pairs (Holm p < 0.05) out of {len(rows)} ===")
    for r in sig_rows:
        print(f"  {r['model_a']} vs {r['model_b']}: Δacc={r['accuracy_diff']:+.4f}, better={r['better_model']}, holm_p={r['holm_p']:.4e}")


if __name__ == "__main__":
    main()
