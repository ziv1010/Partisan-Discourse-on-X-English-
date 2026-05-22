#!/usr/bin/env python3
"""
Backfill stance columns in an extracted keyword CSV from an existing combined
stance CSV so previously inferred tweets do not need to be re-run.

Matching logic
- Match on normalized `(keyword, tweet text)`.
- The combined CSV contributes its `tweet` field.
- The extracted CSV is checked against `en_text`, `tweet_original`, and
  `tweet`, in that order, because Hindi rows often store the translated text in
  `en_text` while the original tweet text lives in `tweet_original`.

Safety rules
- Only fill empty stance columns unless `--overwrite` is passed.
- If the same normalized key appears in the combined CSV with conflicting
  stance payloads, that key is treated as ambiguous and is skipped.
- If an extracted row matches multiple distinct combined payloads through
  different text fields, that row is also skipped.

Outputs
- A merged CSV with stance fields backfilled where safe.
- A remaining-only CSV containing rows that still have no stance data.
- A text summary with overlap statistics.
"""

from __future__ import annotations

import argparse
import csv
import html
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from re import compile as re_compile


BASE_DIR = Path(
    "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Additional_Steps/Crosschecking_and_Joining"
)
DEFAULT_EXTRACTED_CSV = BASE_DIR / "extracted_by_keyword" / "extracted_ALL_35keywords_english_hindi_only.csv"
DEFAULT_COMBINED_CSV = BASE_DIR / "combined_stance_results.csv"

STANCE_COLUMNS = [
    "subjects_scored",
    "stance_gold",
    "fewshot_raw",
    "fewshot_label",
    "fewshot_label_for_against",
    "fewshot_reason",
    "fewshot_shots_json",
]

TEXT_FIELDS_TO_MATCH = ["en_text", "tweet_original", "tweet"]
NON_ALNUM_RE = re_compile(r"[^a-z0-9]+")


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


def norm_nospace(value: object) -> str:
    return NON_ALNUM_RE.sub("", clean_text(value).lower())


def stance_payload(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(clean_text(row.get(column) or "") for column in STANCE_COLUMNS)


def has_any_stance(payload: tuple[str, ...]) -> bool:
    return any(payload)


def row_has_any_stance(row: dict[str, str]) -> bool:
    return any(clean_text(row.get(column) or "") for column in STANCE_COLUMNS)


def payload_to_row(payload: tuple[str, ...]) -> dict[str, str]:
    return dict(zip(STANCE_COLUMNS, payload))


def make_key(keyword: str, text: str) -> tuple[str, str] | None:
    keyword_norm = clean_text(keyword).lower()
    text_norm = norm_nospace(text)
    if not keyword_norm or not text_norm:
        return None
    return keyword_norm, text_norm


def default_output_paths(extracted_csv: Path) -> tuple[Path, Path, Path]:
    stem = extracted_csv.stem
    merged_csv = extracted_csv.with_name(f"{stem}_with_existing_stance.csv")
    remaining_csv = extracted_csv.with_name(f"{stem}_remaining_for_inference.csv")
    summary_txt = extracted_csv.with_name(f"{stem}_backfill_summary.txt")
    return merged_csv, remaining_csv, summary_txt


def load_combined_index(combined_csv: Path) -> tuple[dict[tuple[str, str], tuple[str, ...]], Counter]:
    stats: Counter[str] = Counter()
    index: dict[tuple[str, str], tuple[str, ...]] = {}
    conflicting_keys: set[tuple[str, str]] = set()

    with combined_csv.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            stats["combined_rows_scanned"] += 1

            key = make_key(row.get("keyword") or "", row.get("tweet") or "")
            if key is None:
                stats["combined_rows_missing_key"] += 1
                continue
            if key in conflicting_keys:
                stats["combined_rows_skipped_after_conflict"] += 1
                continue

            payload = stance_payload(row)
            if not has_any_stance(payload):
                stats["combined_rows_missing_stance"] += 1
                continue

            existing = index.get(key)
            if existing is None:
                index[key] = payload
                stats["combined_unique_keys_loaded"] += 1
                continue

            if existing == payload:
                stats["combined_duplicate_same_payload"] += 1
                continue

            conflicting_keys.add(key)
            del index[key]
            stats["combined_conflicting_keys"] += 1

    stats["combined_keys_removed_as_conflicting"] = len(conflicting_keys)
    return index, stats


def resolve_payload_for_extracted_row(
    row: dict[str, str],
    combined_index: dict[tuple[str, str], tuple[str, ...]],
    stats: Counter,
) -> tuple[str, ...] | None:
    keyword = row.get("keyword") or ""
    matched_payloads: list[tuple[str, ...]] = []
    seen_keys: set[tuple[str, str]] = set()

    for field in TEXT_FIELDS_TO_MATCH:
        key = make_key(keyword, row.get(field) or "")
        if key is None or key in seen_keys:
            continue
        seen_keys.add(key)

        payload = combined_index.get(key)
        if payload is None:
            continue

        matched_payloads.append(payload)

    if not matched_payloads:
        return None

    first_payload = matched_payloads[0]
    if all(payload == first_payload for payload in matched_payloads[1:]):
        return first_payload

    stats["extracted_rows_ambiguous_across_text_fields"] += 1
    return None


def apply_payload_to_row(row: dict[str, str], payload: tuple[str, ...], overwrite: bool, stats: Counter) -> dict[str, str]:
    payload_map = payload_to_row(payload)
    row_updated = dict(row)

    for column in STANCE_COLUMNS:
        incoming = payload_map[column]
        existing = clean_text(row_updated.get(column) or "")

        if overwrite:
            if existing != incoming:
                stats["stance_cells_overwritten"] += 1
            row_updated[column] = incoming
            continue

        if not existing and incoming:
            row_updated[column] = incoming
            stats["stance_cells_filled"] += 1
        elif existing and incoming and existing != incoming:
            stats["stance_cells_left_unchanged_due_to_existing_value"] += 1

    return row_updated


def write_outputs(
    extracted_csv: Path,
    merged_csv: Path,
    remaining_csv: Path,
    combined_index: dict[tuple[str, str], tuple[str, ...]],
    overwrite: bool,
) -> Counter:
    stats: Counter[str] = Counter()

    with extracted_csv.open(newline="", encoding="utf-8") as infile, \
        merged_csv.open("w", newline="", encoding="utf-8") as merged_out, \
        remaining_csv.open("w", newline="", encoding="utf-8") as remaining_out:

        reader = csv.DictReader(infile)
        fieldnames = list(reader.fieldnames or [])
        for column in STANCE_COLUMNS:
            if column not in fieldnames:
                fieldnames.append(column)

        merged_writer = csv.DictWriter(merged_out, fieldnames=fieldnames)
        remaining_writer = csv.DictWriter(remaining_out, fieldnames=fieldnames)
        merged_writer.writeheader()
        remaining_writer.writeheader()

        for row in reader:
            stats["extracted_rows_scanned"] += 1
            original_has_stance = row_has_any_stance(row)
            if original_has_stance:
                stats["extracted_rows_with_preexisting_stance"] += 1

            payload = resolve_payload_for_extracted_row(row, combined_index, stats)
            row_out = dict(row)

            if payload is not None:
                stats["extracted_rows_with_match"] += 1
                row_out = apply_payload_to_row(row_out, payload, overwrite=overwrite, stats=stats)
                if not original_has_stance and row_has_any_stance(row_out):
                    stats["extracted_rows_backfilled"] += 1
            else:
                stats["extracted_rows_without_match"] += 1

            merged_writer.writerow(row_out)

            if row_has_any_stance(row_out):
                stats["merged_rows_with_stance"] += 1
            else:
                remaining_writer.writerow(row_out)
                stats["remaining_rows_without_stance"] += 1

    return stats


def build_summary(
    extracted_csv: Path,
    combined_csv: Path,
    merged_csv: Path,
    remaining_csv: Path,
    overwrite: bool,
    combined_stats: Counter,
    extracted_stats: Counter,
) -> str:
    lines = [
        "EXISTING STANCE BACKFILL SUMMARY",
        f"Combined CSV: {combined_csv}",
        f"Extracted CSV: {extracted_csv}",
        f"Merged output CSV: {merged_csv}",
        f"Remaining-only CSV: {remaining_csv}",
        f"Overwrite existing stance fields: {'yes' if overwrite else 'no'}",
        "",
        "Combined index",
        f"- Rows scanned: {combined_stats['combined_rows_scanned']:,}",
        f"- Rows missing key: {combined_stats['combined_rows_missing_key']:,}",
        f"- Rows missing stance payload: {combined_stats['combined_rows_missing_stance']:,}",
        f"- Unique matchable keys loaded: {combined_stats['combined_unique_keys_loaded']:,}",
        f"- Duplicate rows with same payload: {combined_stats['combined_duplicate_same_payload']:,}",
        f"- Conflicting keys removed: {combined_stats['combined_keys_removed_as_conflicting']:,}",
        "",
        "Extracted processing",
        f"- Rows scanned: {extracted_stats['extracted_rows_scanned']:,}",
        f"- Rows with pre-existing stance: {extracted_stats['extracted_rows_with_preexisting_stance']:,}",
        f"- Rows with at least one combined match: {extracted_stats['extracted_rows_with_match']:,}",
        f"- Rows backfilled from empty stance: {extracted_stats['extracted_rows_backfilled']:,}",
        f"- Rows skipped due to ambiguous text-field matches: {extracted_stats['extracted_rows_ambiguous_across_text_fields']:,}",
        f"- Rows without a usable combined match: {extracted_stats['extracted_rows_without_match']:,}",
        "",
        "Output",
        f"- Merged rows containing stance data: {extracted_stats['merged_rows_with_stance']:,}",
        f"- Remaining rows still lacking stance data: {extracted_stats['remaining_rows_without_stance']:,}",
        f"- Stance cells filled: {extracted_stats['stance_cells_filled']:,}",
        f"- Stance cells overwritten: {extracted_stats['stance_cells_overwritten']:,}",
        f"- Existing non-empty cells left unchanged: {extracted_stats['stance_cells_left_unchanged_due_to_existing_value']:,}",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted-csv", type=Path, default=DEFAULT_EXTRACTED_CSV)
    parser.add_argument("--combined-csv", type=Path, default=DEFAULT_COMBINED_CSV)
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Merged CSV path. Defaults to '<input>_with_existing_stance.csv'.",
    )
    parser.add_argument(
        "--remaining-csv",
        type=Path,
        help="Rows that still have no stance data after backfill.",
    )
    parser.add_argument(
        "--summary-txt",
        type=Path,
        help="Optional path for the run summary text file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite non-empty stance cells in the extracted CSV with combined values.",
    )
    return parser.parse_args()


def main() -> None:
    set_large_csv_limit()
    args = parse_args()

    merged_csv, remaining_csv, summary_txt = default_output_paths(args.extracted_csv)
    if args.output_csv is not None:
        merged_csv = args.output_csv
    if args.remaining_csv is not None:
        remaining_csv = args.remaining_csv
    if args.summary_txt is not None:
        summary_txt = args.summary_txt

    output_paths = [merged_csv.resolve(), remaining_csv.resolve(), summary_txt.resolve()]
    if len(set(output_paths)) != len(output_paths):
        raise SystemExit("Output paths must be distinct.")
    if merged_csv.resolve() == args.extracted_csv.resolve() or remaining_csv.resolve() == args.extracted_csv.resolve():
        raise SystemExit("Choose output paths different from the extracted input CSV.")

    combined_index, combined_stats = load_combined_index(args.combined_csv)
    extracted_stats = write_outputs(
        extracted_csv=args.extracted_csv,
        merged_csv=merged_csv,
        remaining_csv=remaining_csv,
        combined_index=combined_index,
        overwrite=args.overwrite,
    )

    summary = build_summary(
        extracted_csv=args.extracted_csv,
        combined_csv=args.combined_csv,
        merged_csv=merged_csv,
        remaining_csv=remaining_csv,
        overwrite=args.overwrite,
        combined_stats=combined_stats,
        extracted_stats=extracted_stats,
    )
    summary_txt.write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()
