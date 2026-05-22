#!/usr/bin/env python3
"""
Extract per-keyword accuracy and F1 metrics for the normal and temporal
fine-tuned Mistral LoRA stance models.

Inputs are existing evaluation artifacts:
- Normal fine-tuned model: 4_b_TestingComparison/results/evaluation/per_keyword_metrics.csv
- Temporal fine-tuned model: 4_f_temporal/.../evaluation/per_keyword_metrics.csv
"""

from __future__ import annotations

import csv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
FINETUNING_DIR = BASE_DIR.parent

NORMAL_METRICS = (
    FINETUNING_DIR
    / "4_b_TestingComparison"
    / "results"
    / "evaluation"
    / "per_keyword_metrics.csv"
)
TEMPORAL_METRICS = (
    FINETUNING_DIR
    / "4_f_temporal"
    / "runs"
    / "temporal_lora_2020_2021_test_2022_2023_gpus_5_7"
    / "evaluation"
    / "per_keyword_metrics.csv"
)

LONG_OUT = BASE_DIR / "finetuned_models_per_keyword_accuracy_f1_long.csv"
WIDE_OUT = BASE_DIR / "finetuned_models_per_keyword_accuracy_f1_wide.csv"
README_OUT = BASE_DIR / "README.md"


def clean_keyword(value: object) -> str:
    return str(value or "").strip().lower()


def float_or_blank(value: object) -> str:
    text = str(value or "").strip()
    if text == "":
        return ""
    return f"{float(text):.6f}"


def int_or_blank(value: object) -> str:
    text = str(value or "").strip()
    if text == "":
        return ""
    return str(int(float(text)))


def read_normal_metrics() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with NORMAL_METRICS.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "model": "normal_finetuned_mistral_lora",
                    "experiment": "normal",
                    "keyword": clean_keyword(row.get("keyword")),
                    "samples": int_or_blank(row.get("count")),
                    "unique_tweets": "",
                    "correct": "",
                    "accuracy": float_or_blank(row.get("accuracy")),
                    "f1_macro": float_or_blank(row.get("f1_macro")),
                    "f1_weighted": "",
                    "precision_macro": "",
                    "recall_macro": "",
                    "gold_for": "",
                    "gold_against": "",
                    "gold_neutral": "",
                    "pred_for": "",
                    "pred_against": "",
                    "pred_neutral": "",
                    "source_file": str(NORMAL_METRICS),
                }
            )
    return rows


def read_temporal_metrics() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with TEMPORAL_METRICS.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "model": "temporal_finetuned_mistral_lora",
                    "experiment": "temporal_2020_2021_train_2022_2023_test",
                    "keyword": clean_keyword(row.get("keyword")),
                    "samples": int_or_blank(row.get("samples")),
                    "unique_tweets": int_or_blank(row.get("unique_tweets")),
                    "correct": int_or_blank(row.get("correct")),
                    "accuracy": float_or_blank(row.get("accuracy")),
                    "f1_macro": float_or_blank(row.get("f1_macro")),
                    "f1_weighted": float_or_blank(row.get("f1_weighted")),
                    "precision_macro": float_or_blank(row.get("precision_macro")),
                    "recall_macro": float_or_blank(row.get("recall_macro")),
                    "gold_for": int_or_blank(row.get("gold_for")),
                    "gold_against": int_or_blank(row.get("gold_against")),
                    "gold_neutral": int_or_blank(row.get("gold_neutral")),
                    "pred_for": int_or_blank(row.get("pred_for")),
                    "pred_against": int_or_blank(row.get("pred_against")),
                    "pred_neutral": int_or_blank(row.get("pred_neutral")),
                    "source_file": str(TEMPORAL_METRICS),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_wide_rows(long_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_keyword: dict[str, dict[str, dict[str, str]]] = {}
    for row in long_rows:
        by_keyword.setdefault(row["keyword"], {})[row["experiment"]] = row

    rows: list[dict[str, str]] = []
    for keyword in sorted(by_keyword):
        normal = by_keyword[keyword].get("normal", {})
        temporal = by_keyword[keyword].get("temporal_2020_2021_train_2022_2023_test", {})
        rows.append(
            {
                "keyword": keyword,
                "normal_samples": normal.get("samples", ""),
                "normal_accuracy": normal.get("accuracy", ""),
                "normal_f1_macro": normal.get("f1_macro", ""),
                "temporal_samples": temporal.get("samples", ""),
                "temporal_accuracy": temporal.get("accuracy", ""),
                "temporal_f1_macro": temporal.get("f1_macro", ""),
                "temporal_f1_weighted": temporal.get("f1_weighted", ""),
                "accuracy_delta_temporal_minus_normal": metric_delta(
                    temporal.get("accuracy", ""), normal.get("accuracy", "")
                ),
                "f1_macro_delta_temporal_minus_normal": metric_delta(
                    temporal.get("f1_macro", ""), normal.get("f1_macro", "")
                ),
            }
        )
    return rows


def metric_delta(left: str, right: str) -> str:
    if left == "" or right == "":
        return ""
    return f"{float(left) - float(right):.6f}"


def write_readme(long_rows: list[dict[str, str]]) -> None:
    normal_n = sum(int(row["samples"]) for row in long_rows if row["experiment"] == "normal")
    temporal_n = sum(
        int(row["samples"])
        for row in long_rows
        if row["experiment"] == "temporal_2020_2021_train_2022_2023_test"
    )
    text = "\n".join(
        [
            "# Fine-Tuned Model Per-Keyword Metrics",
            "",
            "Generated combined per-keyword accuracy and macro-F1 files for the two fine-tuned Mistral LoRA models.",
            "",
            "## Outputs",
            "",
            "- `finetuned_models_per_keyword_accuracy_f1_long.csv`: one row per model-keyword.",
            "- `finetuned_models_per_keyword_accuracy_f1_wide.csv`: one row per keyword, with normal and temporal columns side by side.",
            "",
            "## Sources",
            "",
            f"- Normal: `{NORMAL_METRICS}`",
            f"- Temporal: `{TEMPORAL_METRICS}`",
            "",
            "## Row Accounting",
            "",
            f"- Normal fine-tuned model: {normal_n:,} evaluated samples across 15 keywords.",
            f"- Temporal fine-tuned model: {temporal_n:,} evaluated samples across 35 keywords.",
            "",
            "Note: the normal evaluation file contains only `count`, `accuracy`, and `f1_macro` per keyword. The temporal evaluation includes additional fields such as weighted F1, precision, recall, and gold/predicted class counts; these are preserved in the long CSV.",
            "",
            "Regenerate with:",
            "",
            "```bash",
            "python3 extract_finetuned_keyword_metrics.py",
            "```",
            "",
        ]
    )
    README_OUT.write_text(text, encoding="utf-8")


def main() -> None:
    normal_rows = read_normal_metrics()
    temporal_rows = read_temporal_metrics()
    long_rows = sorted(normal_rows + temporal_rows, key=lambda row: (row["keyword"], row["experiment"]))

    long_fields = [
        "model",
        "experiment",
        "keyword",
        "samples",
        "unique_tweets",
        "correct",
        "accuracy",
        "f1_macro",
        "f1_weighted",
        "precision_macro",
        "recall_macro",
        "gold_for",
        "gold_against",
        "gold_neutral",
        "pred_for",
        "pred_against",
        "pred_neutral",
        "source_file",
    ]
    wide_fields = [
        "keyword",
        "normal_samples",
        "normal_accuracy",
        "normal_f1_macro",
        "temporal_samples",
        "temporal_accuracy",
        "temporal_f1_macro",
        "temporal_f1_weighted",
        "accuracy_delta_temporal_minus_normal",
        "f1_macro_delta_temporal_minus_normal",
    ]

    write_csv(LONG_OUT, long_rows, long_fields)
    write_csv(WIDE_OUT, make_wide_rows(long_rows), wide_fields)
    write_readme(long_rows)

    print(f"Normal rows: {len(normal_rows)} keywords, {sum(int(row['samples']) for row in normal_rows):,} samples")
    print(f"Temporal rows: {len(temporal_rows)} keywords, {sum(int(row['samples']) for row in temporal_rows):,} samples")
    print(f"Saved: {LONG_OUT}")
    print(f"Saved: {WIDE_OUT}")


if __name__ == "__main__":
    main()
