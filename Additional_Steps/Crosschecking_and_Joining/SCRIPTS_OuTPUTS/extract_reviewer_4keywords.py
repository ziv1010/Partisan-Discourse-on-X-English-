#!/usr/bin/env python3
"""
Extract the reviewer-focused 4-keyword subset using the same rules as
keyword_extractor_v2.py.

Selected keywords:
- Dravidian
- HindiImposition
- NEET
- Federalism

Outputs:
- extracted_reviewer_4keywords.csv
- extracted_reviewer_4keywords_ids.txt
- extracted_reviewer_4keywords_summary.csv
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import keyword_extractor_v2 as base


SELECTED_KEYWORDS = [
    "Dravidian",
    "HindiImposition",
    "NEET",
    "Federalism",
]

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_DIR / "extracted_by_keyword"

COMBINED_CSV = OUT_DIR / "extracted_reviewer_4keywords.csv"
COMBINED_IDS = OUT_DIR / "extracted_reviewer_4keywords_ids.txt"
SUMMARY_CSV = OUT_DIR / "extracted_reviewer_4keywords_summary.csv"


def selected_keyword_specs() -> list[tuple[str, list[str]]]:
    specs = []
    for keyword in SELECTED_KEYWORDS:
        variants = [base.norm_nospace(variant) for variant in base.phrase_variants(keyword)]
        variants = [variant for variant in variants if variant]
        if variants:
            specs.append((keyword, variants))
    return specs


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    keyword_defs = selected_keyword_specs()
    keyword_to_seen_keys = {keyword: set() for keyword, _ in keyword_defs}
    stats = {
        keyword: {
            "total_extracted": 0,
            "by_label": Counter(),
        }
        for keyword, _ in keyword_defs
    }
    translation_stats = Counter()
    total_rows_scanned = 0
    total_rows_written = 0

    print(f"CSV Path: {base.CSV_PATH}")
    print(f"Output CSV: {COMBINED_CSV}")
    print(f"Search column: {base.SEARCH_TEXT_COL}")
    print(f"Cosine filter: translated rows with {base.COSINE_COL} >= {base.MIN_COSINE:.2f}")
    print(f"Keywords: {', '.join(SELECTED_KEYWORDS)}")
    print("Mode: EN_TEXT SEARCH ONLY")
    print("Dedup Rule: Same en_text + same author NOT allowed within same keyword")

    with base.CSV_PATH.open(newline="", encoding="utf-8") as infile, \
        COMBINED_CSV.open("w", newline="", encoding="utf-8") as combined_out, \
        COMBINED_IDS.open("w", encoding="utf-8") as combined_ids_out:

        reader = csv.DictReader(infile)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header row.")
        if base.SEARCH_TEXT_COL not in reader.fieldnames:
            raise ValueError(
                f"Required column '{base.SEARCH_TEXT_COL}' not found. "
                f"Available columns: {reader.fieldnames}"
            )

        combined_writer = csv.DictWriter(combined_out, fieldnames=base.OUTPUT_COLUMNS)
        combined_writer.writeheader()

        for row_index, row in enumerate(reader):
            total_rows_scanned += 1

            source_row = base.clean_text(row.get("row_id") or row.get("source_row") or row_index)
            tweet_original = base.clean_text(row.get("tweet_original") or row.get("tweet") or "")
            en_text = base.clean_text(row.get(base.SEARCH_TEXT_COL) or "")
            if not en_text:
                continue

            tweet_label, label_norm = base.map_party_labels(row.get("retweet_party") or "")
            if label_norm not in base.TARGETS:
                continue

            cosine = base.parse_float(row.get(base.COSINE_COL) or "")
            if cosine is None:
                if base.is_unchanged_english_row(row, tweet_original, en_text):
                    translation_stats["blank_cosine_kept_as_english"] += 1
                else:
                    translation_stats["blank_cosine_filtered"] += 1
                    continue
            elif cosine < base.MIN_COSINE:
                translation_stats["low_cosine_filtered"] += 1
                continue
            else:
                translation_stats["cosine_kept"] += 1

            search_text_norm = base.norm_nospace(en_text)
            if not search_text_norm:
                continue

            author = base.choose_author(row)
            tweet_author_key = f"{search_text_norm}|||{author.lower()}"

            matched_keywords = []
            for keyword, variants in keyword_defs:
                if any(variant in search_text_norm for variant in variants):
                    matched_keywords.append(keyword)

            if not matched_keywords:
                continue

            for keyword in matched_keywords:
                if tweet_author_key in keyword_to_seen_keys[keyword]:
                    continue

                keyword_to_seen_keys[keyword].add(tweet_author_key)
                output_row = base.make_output_row(
                    row=row,
                    keyword=keyword,
                    source_row=source_row,
                    en_text=en_text,
                    tweet_label=tweet_label,
                    label_norm=label_norm,
                )

                combined_writer.writerow(output_row)
                combined_ids_out.write(f"{source_row}\n")

                stats[keyword]["total_extracted"] += 1
                stats[keyword]["by_label"][label_norm] += 1
                total_rows_written += 1

            if total_rows_scanned % 100000 == 0:
                print(
                    f"Scanned {total_rows_scanned:,} rows | "
                    f"written {total_rows_written:,} keyword-matched rows"
                )

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as summary_out:
        fieldnames = ["keyword", "total", "pro_ruling", "pro_opposition"]
        writer = csv.DictWriter(summary_out, fieldnames=fieldnames)
        writer.writeheader()

        print("\nSUMMARY")
        print(f"Rows scanned: {total_rows_scanned:,}")
        print(f"Rows written: {total_rows_written:,}")
        print(f"Blank cosine kept as unchanged English rows: {translation_stats['blank_cosine_kept_as_english']:,}")
        print(f"Blank cosine filtered: {translation_stats['blank_cosine_filtered']:,}")
        print(f"Low cosine filtered: {translation_stats['low_cosine_filtered']:,}")
        print(f"Rows kept with numeric cosine >= threshold: {translation_stats['cosine_kept']:,}")

        for keyword, _variants in keyword_defs:
            row = {
                "keyword": keyword,
                "total": stats[keyword]["total_extracted"],
                "pro_ruling": stats[keyword]["by_label"].get("pro ruling", 0),
                "pro_opposition": stats[keyword]["by_label"].get("pro opposition", 0),
            }
            writer.writerow(row)
            print(
                f"{keyword:18} total={row['total']:6} | "
                f"pro_ruling={row['pro_ruling']:6} | "
                f"pro_opposition={row['pro_opposition']:6}"
            )

    print("\nFiles saved:")
    print(f"- Combined CSV: {COMBINED_CSV}")
    print(f"- Combined IDs: {COMBINED_IDS}")
    print(f"- Summary CSV: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
