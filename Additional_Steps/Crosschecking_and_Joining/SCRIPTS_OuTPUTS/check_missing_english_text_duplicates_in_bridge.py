#!/usr/bin/env python3
"""
For combined rows that were missing from the extracted keyword CSV and also not
found in the bridge CSV by exact source_row -> row_id, check whether their text
still appears somewhere in the bridge CSV under a different row_id.

Inputs
- coverage_missing_vs_either.csv
- retweet_data_combined_labeled_by_retweet_en_with_hinglish_bridge.csv

Outputs
- english_missing_not_in_bridge_by_id_but_found_by_text.csv
- english_missing_not_in_bridge_and_not_found_by_text.csv
"""

from __future__ import annotations

import csv
import html
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


BASE_DIR = Path(
    "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Additional_Steps/Crosschecking_and_Joining"
)
MISSING_CSV = BASE_DIR / "coverage_missing_vs_either.csv"
BRIDGE_CSV = BASE_DIR / "retweet_data_combined_labeled_by_retweet_en_with_hinglish_bridge.csv"

FOUND_BY_TEXT_CSV = BASE_DIR / "english_missing_not_in_bridge_by_id_but_found_by_text.csv"
NOT_FOUND_BY_TEXT_CSV = BASE_DIR / "english_missing_not_in_bridge_and_not_found_by_text.csv"

TEXT_FIELDS = [
    "tweet",
    "en_text",
    "tweet_original",
    "hinglish_english_text",
    "hinglish_hindi_text",
    "hinglish_roman_text",
]
EXTRACTION_FIELDS = {"en_text", "tweet_original"}

NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def clean_text(value: object) -> str:
    text = html.unescape(str(value or ""))
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    return text.strip()


def norm_nospace(value: object) -> str:
    return NON_ALNUM_RE.sub("", clean_text(value).lower())


def load_missing_english_rows() -> tuple[list[dict[str, str]], dict[str, list[int]], set[str]]:
    rows: list[dict[str, str]] = []
    by_id: dict[str, list[int]] = defaultdict(list)
    norms: set[str] = set()

    with MISSING_CSV.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            language = clean_text(row.get("language") or "").lower()
            if language != "english":
                continue

            tweet = clean_text(row.get("tweet") or "")
            tweet_norm = norm_nospace(tweet)
            if not tweet_norm:
                continue

            item = {
                "source_row": clean_text(row.get("source_row") or ""),
                "keyword": clean_text(row.get("keyword") or "").lower(),
                "language": language,
                "tweet_label": clean_text(row.get("tweet_label") or ""),
                "original_author": clean_text(row.get("original_author") or ""),
                "tweet": tweet,
                "tweet_norm": tweet_norm,
            }
            idx = len(rows)
            rows.append(item)
            if item["source_row"]:
                by_id[item["source_row"]].append(idx)
            norms.add(tweet_norm)

    return rows, by_id, norms


def scan_bridge(
    missing_english_by_id: dict[str, list[int]],
    missing_english_norms: set[str],
) -> tuple[set[str], dict[str, dict[str, object]]]:
    found_by_id: set[str] = set()
    text_match_info: dict[str, dict[str, object]] = {}

    with BRIDGE_CSV.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for rows_scanned, row in enumerate(reader, 1):
            row_id = clean_text(row.get("row_id") or "")
            if row_id in missing_english_by_id:
                found_by_id.add(row_id)

            for field in TEXT_FIELDS:
                text_norm = norm_nospace(row.get(field) or "")
                if not text_norm or text_norm not in missing_english_norms:
                    continue

                info = text_match_info.setdefault(text_norm, {"fields": set(), "row_ids": []})
                info["fields"].add(field)
                if row_id and row_id not in info["row_ids"] and len(info["row_ids"]) < 10:
                    info["row_ids"].append(row_id)

            if rows_scanned % 1_000_000 == 0:
                print(
                    f"Scanned {rows_scanned:,} bridge rows | "
                    f"row_id matches {len(found_by_id):,} / {len(missing_english_by_id):,} | "
                    f"text matches {len(text_match_info):,}"
                )

    return found_by_id, text_match_info


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    missing_english_rows, missing_english_by_id, missing_english_norms = load_missing_english_rows()
    found_by_id, text_match_info = scan_bridge(missing_english_by_id, missing_english_norms)

    rows_to_check = []
    for item in missing_english_rows:
        if item["source_row"] and item["source_row"] in found_by_id:
            continue
        rows_to_check.append(item)

    summary = Counter()
    field_combo_counts = Counter()
    keyword_found_counts = Counter()
    keyword_not_found_counts = Counter()
    found_rows = []
    not_found_rows = []

    for item in rows_to_check:
        info = text_match_info.get(item["tweet_norm"])
        summary["rows_not_found_by_id"] += 1

        if info:
            fields = sorted(info["fields"])
            sample_row_ids = info["row_ids"]
            found_in_extraction_fields = bool(EXTRACTION_FIELDS.intersection(info["fields"]))
            found_only_outside_extraction_fields = not found_in_extraction_fields

            out = {
                "source_row": item["source_row"],
                "keyword": item["keyword"],
                "language": item["language"],
                "tweet_label": item["tweet_label"],
                "original_author": item["original_author"],
                "tweet": item["tweet"],
                "matched_fields": "|".join(fields),
                "sample_bridge_row_ids": "|".join(sample_row_ids),
                "found_in_extraction_fields": "yes" if found_in_extraction_fields else "no",
                "found_only_outside_extraction_fields": "yes" if found_only_outside_extraction_fields else "no",
            }
            found_rows.append(out)

            summary["rows_found_by_text_any_field"] += 1
            if found_in_extraction_fields:
                summary["rows_found_by_text_in_extraction_fields"] += 1
            if found_only_outside_extraction_fields:
                summary["rows_found_only_outside_extraction_fields"] += 1

            field_combo_counts["|".join(fields)] += 1
            keyword_found_counts[item["keyword"]] += 1
        else:
            out = {
                "source_row": item["source_row"],
                "keyword": item["keyword"],
                "language": item["language"],
                "tweet_label": item["tweet_label"],
                "original_author": item["original_author"],
                "tweet": item["tweet"],
            }
            not_found_rows.append(out)
            summary["rows_not_found_by_text_any_field"] += 1
            keyword_not_found_counts[item["keyword"]] += 1

    write_rows(
        FOUND_BY_TEXT_CSV,
        found_rows,
        [
            "source_row",
            "keyword",
            "language",
            "tweet_label",
            "original_author",
            "tweet",
            "matched_fields",
            "sample_bridge_row_ids",
            "found_in_extraction_fields",
            "found_only_outside_extraction_fields",
        ],
    )
    write_rows(
        NOT_FOUND_BY_TEXT_CSV,
        not_found_rows,
        [
            "source_row",
            "keyword",
            "language",
            "tweet_label",
            "original_author",
            "tweet",
        ],
    )

    print("SUMMARY")
    print(f"Rows not found by row_id: {summary['rows_not_found_by_id']:,}")
    print(f"Rows found by text in any bridge field: {summary['rows_found_by_text_any_field']:,}")
    print(f"Rows still not found by text in any bridge field: {summary['rows_not_found_by_text_any_field']:,}")
    print(f"Rows found by text in extraction fields: {summary['rows_found_by_text_in_extraction_fields']:,}")
    print(f"Rows found only outside extraction fields: {summary['rows_found_only_outside_extraction_fields']:,}")

    if summary["rows_not_found_by_id"]:
        pct_any = summary["rows_found_by_text_any_field"] / summary["rows_not_found_by_id"] * 100.0
        pct_extraction = summary["rows_found_by_text_in_extraction_fields"] / summary["rows_not_found_by_id"] * 100.0
        print(f"Pct found by text in any bridge field: {pct_any:.2f}%")
        print(f"Pct found by text in extraction fields: {pct_extraction:.2f}%")

    print("\nTop field combinations")
    for combo, count in field_combo_counts.most_common(12):
        print(f"{combo or '<none>'}: {count:,}")

    print("\nTop keywords found by text")
    for keyword, count in keyword_found_counts.most_common(15):
        print(f"{keyword}: {count:,}")

    print("\nTop keywords still not found by text")
    for keyword, count in keyword_not_found_counts.most_common(15):
        print(f"{keyword}: {count:,}")

    print("\nOutputs")
    print(FOUND_BY_TEXT_CSV)
    print(NOT_FOUND_BY_TEXT_CSV)


if __name__ == "__main__":
    main()
