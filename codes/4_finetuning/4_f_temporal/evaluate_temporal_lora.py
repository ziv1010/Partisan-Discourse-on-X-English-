#!/usr/bin/env python3
"""
Evaluate temporal LoRA stance predictions overall, per keyword, and per keyword-year.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


LABELS = ["For", "Against", "Neutral"]


def normalize_stance(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = text.replace("favour", "for").replace("favor", "for")
    text = text.replace("nuetral", "neutral").replace("oppose", "against")
    key = re.sub(r"[^a-z]+", "", text)
    if key in {"for", "positive", "support", "supports", "pro"}:
        return "For"
    if key in {"against", "negative", "anti", "opposed", "opposes"}:
        return "Against"
    if key in {"neutral", "unrelated", "irrelevant"}:
        return "Neutral"
    return ""


def metric_row(df: pd.DataFrame, group_values: dict[str, Any] | None = None) -> dict[str, Any]:
    y_true = df["gold_norm"].tolist()
    y_pred = df["pred_norm"].tolist()
    row = dict(group_values or {})
    row.update(
        {
            "samples": int(len(df)),
            "unique_tweets": int(df["tweet"].nunique()) if "tweet" in df.columns else int(len(df)),
            "correct": int((df["gold_norm"] == df["pred_norm"]).sum()),
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0),
            "precision_macro": precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        }
    )
    for label in LABELS:
        row[f"gold_{label.lower()}"] = int((df["gold_norm"] == label).sum())
        row[f"pred_{label.lower()}"] = int((df["pred_norm"] == label).sum())
    return row


def grouped_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append(metric_row(group, dict(zip(group_cols, keys))))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gold-col", default="stance")
    parser.add_argument("--pred-col", default="lora_stance")
    args = parser.parse_args()

    pred_path = Path(args.pred_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_path, low_memory=False)
    if args.gold_col not in df.columns:
        raise ValueError(f"Missing gold column: {args.gold_col}")
    if args.pred_col not in df.columns:
        raise ValueError(f"Missing prediction column: {args.pred_col}")

    df["gold_norm"] = df[args.gold_col].map(normalize_stance)
    df["pred_norm"] = df[args.pred_col].map(normalize_stance)
    valid = df[df["gold_norm"].isin(LABELS) & df["pred_norm"].isin(LABELS)].copy()
    invalid = df[~(df["gold_norm"].isin(LABELS) & df["pred_norm"].isin(LABELS))].copy()

    if valid.empty:
        raise ValueError("No valid gold/prediction rows after stance normalization.")

    overall = metric_row(valid)
    overall["prediction_file"] = str(pred_path)
    overall["valid_samples"] = int(len(valid))
    overall["invalid_samples"] = int(len(invalid))

    (output_dir / "overall_metrics.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")

    per_keyword = grouped_metrics(valid, ["keyword"])
    per_keyword.to_csv(output_dir / "per_keyword_metrics.csv", index=False)

    if "year" in valid.columns:
        per_keyword_year = grouped_metrics(valid, ["keyword", "year"])
        per_keyword_year.to_csv(output_dir / "per_keyword_year_metrics.csv", index=False)

    report = classification_report(
        valid["gold_norm"],
        valid["pred_norm"],
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv")

    cm = confusion_matrix(valid["gold_norm"], valid["pred_norm"], labels=LABELS)
    pd.DataFrame(cm, index=[f"gold_{x}" for x in LABELS], columns=[f"pred_{x}" for x in LABELS]).to_csv(
        output_dir / "confusion_matrix.csv"
    )

    if not invalid.empty:
        invalid.to_csv(output_dir / "invalid_prediction_rows.csv", index=False)

    print("=" * 88)
    print("Temporal LoRA Overall")
    print("=" * 88)
    print(
        f"n={overall['samples']} accuracy={overall['accuracy']:.4f} "
        f"macro_f1={overall['f1_macro']:.4f} weighted_f1={overall['f1_weighted']:.4f}"
    )
    print("\nPer Keyword")
    print(
        per_keyword[
            ["keyword", "samples", "correct", "accuracy", "f1_macro", "gold_for", "gold_against", "gold_neutral"]
        ].to_string(index=False)
    )
    print(f"\nSaved evaluation files under: {output_dir}")


if __name__ == "__main__":
    main()
