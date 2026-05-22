#!/usr/bin/env python3
"""Utilities for paired McNemar significance tests on stance predictions."""

from __future__ import annotations

import csv
import html
import math
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


LABELS = ("For", "Against", "Neutral")
MODEL_ORDER = (
    "bert",
    "roberta",
    "pyabsa",
    "mistral_base",
    "mistral_fewshot",
    "mistral_lora",
    "mistral_finetuned",
)

GROUND_TRUTH_COLUMNS = (
    "original_stance",
    "stance",
    "stance_gold",
    "gold_stance",
    "true_stance",
    "label",
)

PREDICTION_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "bert": ("bert_prediction",),
    "roberta": ("roberta_prediction",),
    "pyabsa": ("pyabsa_prediction",),
    "mistral_base": ("mistral_base_prediction", "mistral_prediction"),
    "mistral_fewshot": ("fewshot_label", "mistral_prediction", "fewshot_label_for_against"),
    "mistral_lora": ("fewshot_label", "mistral_lora_prediction", "fewshot_label_for_against"),
    "mistral_finetuned": (
        "mistral_finetuned_prediction",
        "fewshot_label",
        "fewshot_label_for_against",
    ),
}
GENERIC_PREDICTION_COLUMNS = ("prediction", "predicted_stance", "pred")


@dataclass(frozen=True)
class PredictionRecord:
    key: str
    row_index: int
    gold: str
    pred: str


@dataclass
class PredictionSet:
    model: str
    path: Path
    gold_col: str
    pred_col: str
    records: Dict[str, PredictionRecord]
    total_rows: int
    invalid_rows: int


def normalize_stance(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower().strip()
    if lowered in {"nan", "none", "null"}:
        return None

    lowered = lowered.replace(".", "")
    lowered = lowered.replace("_", " ").replace("-", " ")
    lowered = re.sub(r"\s+", " ", lowered).strip(" '\"\t\r\n")
    lowered = lowered.replace("favour", "favor")

    mapping = {
        "for": "For",
        "favor": "For",
        "in favor": "For",
        "positive": "For",
        "support": "For",
        "supports": "For",
        "supported": "For",
        "pro": "For",
        "against": "Against",
        "negative": "Against",
        "oppose": "Against",
        "opposes": "Against",
        "opposed": "Against",
        "anti": "Against",
        "neutral": "Neutral",
        "nuetral": "Neutral",
        "unrelated": "Neutral",
        "irrelevant": "Neutral",
        "no stance": "Neutral",
    }
    return mapping.get(lowered)


def infer_model_name(path: Path) -> str:
    parent_name = path.parent.name
    if parent_name in PREDICTION_COLUMNS:
        return parent_name

    stem = path.stem
    if stem.endswith("_predictions"):
        stem = stem[: -len("_predictions")]
    return stem


def model_sort_key(model: str) -> Tuple[int, str]:
    try:
        return (MODEL_ORDER.index(model), model)
    except ValueError:
        return (len(MODEL_ORDER), model)


def _pick_column(fieldnames: Sequence[str], candidates: Iterable[str]) -> Optional[str]:
    field_set = set(fieldnames)
    for candidate in candidates:
        if candidate in field_set:
            return candidate
    return None


def _make_key(row: Dict[str, str], row_index: int, align_on: str) -> str:
    if align_on == "auto":
        if row.get("tweet") and row.get("keyword"):
            return "\x1f".join(
                (
                    _normalize_key_value("tweet", row.get("tweet", "")),
                    _normalize_key_value("keyword", row.get("keyword", "")),
                )
            )
        if row.get("source_row"):
            return _normalize_key_value("source_row", row.get("source_row", ""))
        return str(row_index)
    if align_on == "index":
        return str(row_index)
    columns = [part.strip() for part in align_on.split(",") if part.strip()]
    if not columns:
        raise ValueError("align_on must be 'index' or one or more comma-separated column names")
    missing = [column for column in columns if column not in row]
    if missing:
        raise ValueError(f"Cannot align on missing column(s): {', '.join(missing)}")
    return "\x1f".join(_normalize_key_value(column, row.get(column, "")) for column in columns)


def _normalize_key_value(column: str, value: object) -> str:
    text = html.unescape(str(value or "")).strip()
    if column == "tweet":
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"\s+", " ", text).strip().lower()
    elif column in {"keyword", "matched keyword"}:
        text = re.sub(r"\s+", " ", text).strip().lower()
    elif column == "source_row":
        text = text[:-2] if text.endswith(".0") else text
    return text


def read_prediction_file(path: Path, model: Optional[str] = None, align_on: str = "index") -> PredictionSet:
    model_name = model or infer_model_name(path)
    path = Path(path)
    pred_candidates = PREDICTION_COLUMNS.get(model_name, ()) + GENERIC_PREDICTION_COLUMNS

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} is empty or has no CSV header")
        fieldnames = list(reader.fieldnames)
        gold_col = _pick_column(fieldnames, GROUND_TRUTH_COLUMNS)
        pred_col = _pick_column(fieldnames, pred_candidates)
        if gold_col is None:
            raise ValueError(f"{path} does not contain a supported gold-label column")
        if pred_col is None:
            raise ValueError(
                f"{path} does not contain a supported prediction column for model '{model_name}'"
            )

        records: Dict[str, PredictionRecord] = {}
        seen_keys: Dict[str, int] = {}
        total_rows = 0
        invalid_rows = 0
        for row_index, row in enumerate(reader):
            total_rows += 1
            gold = normalize_stance(row.get(gold_col))
            pred = normalize_stance(row.get(pred_col))
            if gold not in LABELS or pred not in LABELS:
                invalid_rows += 1
                continue
            base_key = _make_key(row, row_index, align_on)
            duplicate_index = seen_keys.get(base_key, 0)
            seen_keys[base_key] = duplicate_index + 1
            key = base_key if duplicate_index == 0 else f"{base_key}\x1e{duplicate_index}"
            records[key] = PredictionRecord(key=key, row_index=row_index, gold=gold, pred=pred)

    return PredictionSet(
        model=model_name,
        path=path,
        gold_col=gold_col,
        pred_col=pred_col,
        records=records,
        total_rows=total_rows,
        invalid_rows=invalid_rows,
    )


def exact_mcnemar_p_value(a_correct_b_wrong: int, a_wrong_b_correct: int) -> float:
    discordant = a_correct_b_wrong + a_wrong_b_correct
    if discordant == 0:
        return 1.0

    tail = min(a_correct_b_wrong, a_wrong_b_correct)
    log_half = math.log(0.5)
    log_terms = [
        math.lgamma(discordant + 1)
        - math.lgamma(k + 1)
        - math.lgamma(discordant - k + 1)
        + discordant * log_half
        for k in range(tail + 1)
    ]
    max_log = max(log_terms)
    lower_tail = math.exp(max_log) * sum(math.exp(term - max_log) for term in log_terms)
    return min(1.0, 2.0 * lower_tail)


def chi_square_mcnemar(a_correct_b_wrong: int, a_wrong_b_correct: int) -> Tuple[float, float]:
    discordant = a_correct_b_wrong + a_wrong_b_correct
    if discordant == 0:
        return 0.0, 1.0

    statistic = (abs(a_correct_b_wrong - a_wrong_b_correct) - 1) ** 2 / discordant
    p_value = math.erfc(math.sqrt(statistic / 2.0))
    return statistic, p_value


def mcnemar_pair(model_a: PredictionSet, model_b: PredictionSet) -> Dict[str, object]:
    both_correct = 0
    both_wrong = 0
    a_correct_b_wrong = 0
    a_wrong_b_correct = 0
    gold_mismatches = 0

    common_keys = sorted(set(model_a.records).intersection(model_b.records), key=_natural_key)
    for key in common_keys:
        a_record = model_a.records[key]
        b_record = model_b.records[key]
        if a_record.gold != b_record.gold:
            gold_mismatches += 1
            continue

        a_correct = a_record.pred == a_record.gold
        b_correct = b_record.pred == b_record.gold
        if a_correct and b_correct:
            both_correct += 1
        elif (not a_correct) and (not b_correct):
            both_wrong += 1
        elif a_correct and not b_correct:
            a_correct_b_wrong += 1
        else:
            a_wrong_b_correct += 1

    n_compared = both_correct + both_wrong + a_correct_b_wrong + a_wrong_b_correct
    exact_p = exact_mcnemar_p_value(a_correct_b_wrong, a_wrong_b_correct)
    chi2_stat, chi2_p = chi_square_mcnemar(a_correct_b_wrong, a_wrong_b_correct)

    if n_compared:
        accuracy_a = (both_correct + a_correct_b_wrong) / n_compared
        accuracy_b = (both_correct + a_wrong_b_correct) / n_compared
        accuracy_diff = accuracy_a - accuracy_b
    else:
        accuracy_a = 0.0
        accuracy_b = 0.0
        accuracy_diff = 0.0

    better_model = ""
    if a_correct_b_wrong > a_wrong_b_correct:
        better_model = model_a.model
    elif a_wrong_b_correct > a_correct_b_wrong:
        better_model = model_b.model

    return {
        "model_a": model_a.model,
        "model_b": model_b.model,
        "n_compared": n_compared,
        "common_valid_rows": len(common_keys),
        "gold_mismatches": gold_mismatches,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "discordant": a_correct_b_wrong + a_wrong_b_correct,
        "accuracy_a": accuracy_a,
        "accuracy_b": accuracy_b,
        "accuracy_diff_a_minus_b": accuracy_diff,
        "better_model": better_model,
        "exact_p": exact_p,
        "chi2_cc": chi2_stat,
        "chi2_cc_p": chi2_p,
        "model_a_file": str(model_a.path),
        "model_b_file": str(model_b.path),
    }


def pairwise_mcnemar(predictions: Sequence[PredictionSet]) -> List[Dict[str, object]]:
    ordered = sorted(predictions, key=lambda item: model_sort_key(item.model))
    return [mcnemar_pair(model_a, model_b) for model_a, model_b in combinations(ordered, 2)]


def add_holm_correction(
    rows: List[Dict[str, object]],
    group_columns: Sequence[str] = (),
    p_column: str = "exact_p",
    alpha: float = 0.05,
) -> None:
    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in rows:
        key = tuple(row.get(column) for column in group_columns)
        groups.setdefault(key, []).append(row)

    for group_rows in groups.values():
        ordered = sorted(group_rows, key=lambda row: float(row[p_column]))
        adjusted_values: Dict[int, float] = {}
        running_max = 0.0
        total = len(ordered)
        for rank, row in enumerate(ordered, start=1):
            adjusted = min(1.0, (total - rank + 1) * float(row[p_column]))
            running_max = max(running_max, adjusted)
            adjusted_values[id(row)] = running_max
        for row in group_rows:
            row["holm_p"] = adjusted_values[id(row)]
            row[f"exact_significant_{alpha:g}"] = float(row[p_column]) <= alpha
            row[f"holm_significant_{alpha:g}"] = float(row["holm_p"]) <= alpha


def pooled_rows(by_seed_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in by_seed_rows:
        grouped.setdefault((str(row["model_a"]), str(row["model_b"])), []).append(row)

    pooled: List[Dict[str, object]] = []
    count_columns = (
        "n_compared",
        "common_valid_rows",
        "gold_mismatches",
        "both_correct",
        "both_wrong",
        "a_correct_b_wrong",
        "a_wrong_b_correct",
        "discordant",
    )
    for (model_a, model_b), rows in sorted(grouped.items(), key=lambda item: (model_sort_key(item[0][0]), model_sort_key(item[0][1]))):
        out: Dict[str, object] = {"model_a": model_a, "model_b": model_b}
        for column in count_columns:
            out[column] = sum(int(row.get(column, 0)) for row in rows)

        n_compared = int(out["n_compared"])
        if n_compared:
            out["accuracy_a"] = (int(out["both_correct"]) + int(out["a_correct_b_wrong"])) / n_compared
            out["accuracy_b"] = (int(out["both_correct"]) + int(out["a_wrong_b_correct"])) / n_compared
            out["accuracy_diff_a_minus_b"] = out["accuracy_a"] - out["accuracy_b"]
        else:
            out["accuracy_a"] = 0.0
            out["accuracy_b"] = 0.0
            out["accuracy_diff_a_minus_b"] = 0.0

        a_correct_b_wrong = int(out["a_correct_b_wrong"])
        a_wrong_b_correct = int(out["a_wrong_b_correct"])
        out["better_model"] = ""
        if a_correct_b_wrong > a_wrong_b_correct:
            out["better_model"] = model_a
        elif a_wrong_b_correct > a_correct_b_wrong:
            out["better_model"] = model_b
        out["exact_p"] = exact_mcnemar_p_value(a_correct_b_wrong, a_wrong_b_correct)
        out["chi2_cc"], out["chi2_cc_p"] = chi_square_mcnemar(a_correct_b_wrong, a_wrong_b_correct)
        out["seed_count"] = len(rows)
        out["seeds"] = ",".join(str(row.get("seed", "")) for row in rows)
        pooled.append(out)

    return pooled


def discover_prediction_files(results_dir: Path) -> Dict[str, Path]:
    files: Dict[str, Path] = {}
    for path in sorted(Path(results_dir).glob("*_predictions.csv")):
        model = infer_model_name(path)
        files[model] = path
    return dict(sorted(files.items(), key=lambda item: model_sort_key(item[0])))


def write_csv_rows(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = _fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_csv_value(row.get(key, "")) for key in fieldnames})


def write_markdown_table(path: Path, title: str, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        "McNemar's paired test is computed from the two discordant cells:",
        "`a_correct_b_wrong` and `a_wrong_b_correct`.",
        "The reported exact p-value is the two-sided binomial McNemar p-value; `holm_p` is Holm-corrected within the table/family.",
        "",
    ]

    if not rows:
        lines.extend(["No pairwise model comparisons were available.", ""])
    else:
        table_fields = [
            "model_a",
            "model_b",
            "n_compared",
            "a_correct_b_wrong",
            "a_wrong_b_correct",
            "accuracy_diff_a_minus_b",
            "better_model",
            "exact_p",
            "holm_p",
            "holm_significant_0.05",
        ]
        if "seed" in rows[0]:
            table_fields.insert(0, "seed")
        if "seed_count" in rows[0]:
            table_fields.insert(0, "seed_count")

        lines.append("| " + " | ".join(table_fields) + " |")
        lines.append("| " + " | ".join("---" for _ in table_fields) + " |")
        for row in rows:
            lines.append("| " + " | ".join(_format_markdown_value(row.get(field, "")) for field in table_fields) + " |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _fieldnames(rows: Sequence[Dict[str, object]]) -> List[str]:
    preferred = [
        "seed",
        "seed_count",
        "seeds",
        "model_a",
        "model_b",
        "n_compared",
        "common_valid_rows",
        "gold_mismatches",
        "both_correct",
        "both_wrong",
        "a_correct_b_wrong",
        "a_wrong_b_correct",
        "discordant",
        "accuracy_a",
        "accuracy_b",
        "accuracy_diff_a_minus_b",
        "better_model",
        "exact_p",
        "holm_p",
        "exact_significant_0.05",
        "holm_significant_0.05",
        "chi2_cc",
        "chi2_cc_p",
        "model_a_file",
        "model_b_file",
    ]
    keys = set()
    for row in rows:
        keys.update(row.keys())
    return [field for field in preferred if field in keys] + sorted(keys - set(preferred))


def _format_csv_value(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.10g}"
    return value


def _format_markdown_value(value: object) -> str:
    if isinstance(value, float):
        if abs(value) < 0.0001 and value != 0:
            return f"{value:.3e}"
        return f"{value:.4f}"
    return str(value)


def _natural_key(value: str) -> Tuple[int, object]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)
