#!/usr/bin/env python3
"""
Merge a fully backfilled extracted CSV with one or more inference result CSVs.

The base CSV already contains rows backfilled from `combined_stance_results.csv`.
The result CSVs contain stance predictions for the rows that were still missing.

This script overlays the result rows onto the base CSV using a strict row key and
writes one final CSV with the same row count as the base file.
"""

from __future__ import annotations

import argparse
import csv
import glob
import html
import sys
import unicodedata
from collections import Counter
from pathlib import Path


BASE_DIR = Path(
    "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Additional_Steps/Crosschecking_and_Joining"
)
DEFAULT_BASE_CSV = (
    BASE_DIR
    / "extracted_by_keyword"
    / "extracted_ALL_35keywords_english_hindi_only_with_existing_stance.csv"
)
DEFAULT_RESULTS_DIR = (
    BASE_DIR
    / "PostJoining_stance"
    / "extracted_ALL_35keywords_english_hindi_only_remaining_for_inference_mistral7b_lora_stance_parallel_state"
    / "outputs"
)
DEFAULT_OUTPUT_CSV = (
    BASE_DIR
    / "PostJoining_stance"
    / "extracted_ALL_35keywords_english_hindi_only_final_with_existing_and_inferred_stance.csv"
)
DEFAULT_SUMMARY_TXT = (
    BASE_DIR
    / "PostJoining_stance"
    / "extracted_ALL_35keywords_english_hindi_only_final_with_existing_and_inferred_stance_summary.txt"
)

KEY_COLUMNS = ["source_row", "row_id", "keyword", "tweet"]
STANCE_COLUMNS = [
    "subjects_scored",
    "stance_gold",
    "fewshot_raw",
    "fewshot_label",
    "fewshot_label_for_against",
    "fewshot_reason",
    "fewshot_shots_json",
]


def set_large_csv_limit() -> None:
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)


def clean_text(value: object) -> str:
    text = html.unescape(str(value or ""))
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    return text.strip()


def row_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(clean_text(row.get(column) or "") for column in KEY_COLUMNS)


def has_any_stance(row: dict[str, str]) -> bool:
    return any(clean_text(row.get(column) or "") for column in STANCE_COLUMNS)


def discover_result_csvs(result_csvs: list[str], results_dir: Path) -> list[Path]:
    if result_csvs:
        paths = [Path(path).expanduser().resolve() for path in result_csvs]
    else:
        paths = sorted(Path(path).resolve() for path in glob.glob(str(results_dir / "result_*.csv")))

    if not paths:
        raise SystemExit("No result CSVs found.")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing result CSVs: {missing}")
    return paths


def load_result_rows(result_paths: list[Path]) -> tuple[dict[tuple[str, ...], dict[str, str]], Counter]:
    stats: Counter[str] = Counter()
    result_map: dict[tuple[str, ...], dict[str, str]] = {}

    for path in result_paths:
        with path.open(newline="", encoding="utf-8") as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                stats["result_rows_scanned"] += 1
                key = row_key(row)
                if not any(key):
                    stats["result_rows_missing_key"] += 1
                    continue
                if key in result_map:
                    raise SystemExit(f"Duplicate result key detected in {path}: {key}")
                result_map[key] = row
                stats["result_unique_keys"] += 1

    return result_map, stats


def merge_csvs(
    base_csv: Path,
    result_map: dict[tuple[str, ...], dict[str, str]],
    output_csv: Path,
) -> Counter:
    stats: Counter[str] = Counter()

    with base_csv.open(newline="", encoding="utf-8") as infile, output_csv.open(
        "w", newline="", encoding="utf-8"
    ) as outfile:
        reader = csv.DictReader(infile)
        fieldnames = list(reader.fieldnames or [])
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            stats["base_rows_scanned"] += 1
            key = row_key(row)
            row_out = dict(row)
            base_has_stance = has_any_stance(row)

            result_row = result_map.get(key)
            if result_row is not None:
                stats["base_rows_matched_to_results"] += 1
                for column in STANCE_COLUMNS:
                    row_out[column] = result_row.get(column, row_out.get(column, ""))
                if not base_has_stance and has_any_stance(row_out):
                    stats["rows_filled_from_results"] += 1
            elif not base_has_stance:
                stats["base_rows_still_missing_stance"] += 1

            if has_any_stance(row_out):
                stats["final_rows_with_stance"] += 1

            writer.writerow(row_out)

    stats["final_rows_without_stance"] = stats["base_rows_scanned"] - stats["final_rows_with_stance"]
    return stats


def build_summary(
    base_csv: Path,
    result_paths: list[Path],
    output_csv: Path,
    result_stats: Counter,
    merge_stats: Counter,
) -> str:
    lines = [
        "FINAL STANCE MERGE SUMMARY",
        f"Base CSV: {base_csv}",
        f"Result CSV count: {len(result_paths)}",
        f"Result CSVs: {', '.join(str(path) for path in result_paths)}",
        f"Final output CSV: {output_csv}",
        "",
        "Results loaded",
        f"- Rows scanned from result files: {result_stats['result_rows_scanned']:,}",
        f"- Unique result keys: {result_stats['result_unique_keys']:,}",
        f"- Result rows missing key: {result_stats['result_rows_missing_key']:,}",
        "",
        "Merge outcome",
        f"- Base rows scanned: {merge_stats['base_rows_scanned']:,}",
        f"- Base rows matched to result rows: {merge_stats['base_rows_matched_to_results']:,}",
        f"- Base rows filled from result rows: {merge_stats['rows_filled_from_results']:,}",
        f"- Final rows with stance data: {merge_stats['final_rows_with_stance']:,}",
        f"- Final rows without stance data: {merge_stats['final_rows_without_stance']:,}",
        f"- Base rows still missing stance after merge: {merge_stats['base_rows_still_missing_stance']:,}",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-csv", type=Path, default=DEFAULT_BASE_CSV)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing result_*.csv files.",
    )
    parser.add_argument(
        "--result-csv",
        action="append",
        default=[],
        help="Optional explicit result CSV path. Can be provided multiple times.",
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--summary-txt", type=Path, default=DEFAULT_SUMMARY_TXT)
    return parser.parse_args()


def main() -> None:
    set_large_csv_limit()
    args = parse_args()

    base_csv = args.base_csv.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()
    summary_txt = args.summary_txt.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    result_paths = discover_result_csvs(args.result_csv, results_dir)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_txt.parent.mkdir(parents=True, exist_ok=True)

    result_map, result_stats = load_result_rows(result_paths)
    merge_stats = merge_csvs(base_csv, result_map, output_csv)
    summary = build_summary(base_csv, result_paths, output_csv, result_stats, merge_stats)
    summary_txt.write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()
