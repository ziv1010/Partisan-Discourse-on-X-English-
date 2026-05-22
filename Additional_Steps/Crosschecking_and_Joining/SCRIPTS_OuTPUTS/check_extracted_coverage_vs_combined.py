#!/usr/bin/env python3
"""
Check whether the new extracted keyword CSV covers the rows in
`combined_stance_results.csv`.

Comparison logic
- The combined file contributes one text field: `tweet`.
- The extracted file contributes two comparison fields:
  - `en_text` (English-side text)
  - `tweet_original` (original/Hindi-side text)
- Coverage is reported three ways:
  - combined `tweet` found in extracted `en_text`
  - combined `tweet` found in extracted `tweet_original`
  - combined `tweet` found in either extracted text field

Normalization is case-insensitive and removes punctuation / spacing noise so
small formatting differences do not count as mismatches.

Outputs:
- coverage_missing_vs_en_text.csv
- coverage_missing_vs_tweet_original.csv
- coverage_missing_vs_either.csv
- coverage_summary_by_language.csv
- coverage_summary_by_keyword.csv
- coverage_summary.txt
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
EXTRACTED_CSV = BASE_DIR / "extracted_by_keyword" / "extracted_ALL_37keywords_cosine070.csv"
COMBINED_CSV = BASE_DIR / "combined_stance_results.csv"

MISSING_EN_CSV = BASE_DIR / "coverage_missing_vs_en_text.csv"
MISSING_ORIGINAL_CSV = BASE_DIR / "coverage_missing_vs_tweet_original.csv"
MISSING_EITHER_CSV = BASE_DIR / "coverage_missing_vs_either.csv"
SUMMARY_BY_LANGUAGE_CSV = BASE_DIR / "coverage_summary_by_language.csv"
SUMMARY_BY_KEYWORD_CSV = BASE_DIR / "coverage_summary_by_keyword.csv"
SUMMARY_TXT = BASE_DIR / "coverage_summary.txt"


NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def clean_text(value: object) -> str:
    text = html.unescape(str(value or ""))
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    return text.strip()


def norm_nospace(value: object) -> str:
    return NON_ALNUM_RE.sub("", clean_text(value).lower())


def pct(part: int, whole: int) -> float:
    return (part / whole * 100.0) if whole else 0.0


def load_extracted_indexes() -> tuple[set[str], set[str], Counter, Counter]:
    extracted_en_texts: set[str] = set()
    extracted_original_texts: set[str] = set()
    extracted_keyword_counts: Counter[str] = Counter()
    extracted_language_counts: Counter[str] = Counter()

    with EXTRACTED_CSV.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            keyword = clean_text(row.get("keyword") or "").lower()
            language = clean_text(row.get("language") or "").lower() or "unknown"
            en_text_norm = norm_nospace(row.get("en_text") or row.get("tweet") or "")
            original_text_norm = norm_nospace(row.get("tweet_original") or "")

            if keyword:
                extracted_keyword_counts[keyword] += 1
            extracted_language_counts[language] += 1

            if en_text_norm:
                extracted_en_texts.add(en_text_norm)
            if original_text_norm:
                extracted_original_texts.add(original_text_norm)

    return extracted_en_texts, extracted_original_texts, extracted_keyword_counts, extracted_language_counts


def write_missing_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "source_row",
        "keyword",
        "language",
        "tweet_label",
        "original_author",
        "tweet",
    ]
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_by_language(path: Path, stats: dict[str, Counter]) -> None:
    fieldnames = [
        "language",
        "combined_rows",
        "covered_in_en_text_rows",
        "missing_in_en_text_rows",
        "covered_in_tweet_original_rows",
        "missing_in_tweet_original_rows",
        "covered_in_either_rows",
        "missing_in_either_rows",
    ]
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for language in sorted(stats):
            row = stats[language]
            writer.writerow(
                {
                    "language": language,
                    "combined_rows": row["combined_rows"],
                    "covered_in_en_text_rows": row["covered_in_en_text_rows"],
                    "missing_in_en_text_rows": row["combined_rows"] - row["covered_in_en_text_rows"],
                    "covered_in_tweet_original_rows": row["covered_in_tweet_original_rows"],
                    "missing_in_tweet_original_rows": row["combined_rows"] - row["covered_in_tweet_original_rows"],
                    "covered_in_either_rows": row["covered_in_either_rows"],
                    "missing_in_either_rows": row["combined_rows"] - row["covered_in_either_rows"],
                }
            )


def write_summary_by_keyword(path: Path, stats: dict[str, Counter], extracted_keyword_counts: Counter) -> None:
    fieldnames = [
        "keyword",
        "combined_rows",
        "extracted_rows",
        "covered_in_en_text_rows",
        "missing_in_en_text_rows",
        "covered_in_tweet_original_rows",
        "missing_in_tweet_original_rows",
        "covered_in_either_rows",
        "missing_in_either_rows",
    ]
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for keyword in sorted(set(stats) | set(extracted_keyword_counts)):
            row = stats[keyword]
            writer.writerow(
                {
                    "keyword": keyword,
                    "combined_rows": row["combined_rows"],
                    "extracted_rows": extracted_keyword_counts.get(keyword, 0),
                    "covered_in_en_text_rows": row["covered_in_en_text_rows"],
                    "missing_in_en_text_rows": row["combined_rows"] - row["covered_in_en_text_rows"],
                    "covered_in_tweet_original_rows": row["covered_in_tweet_original_rows"],
                    "missing_in_tweet_original_rows": row["combined_rows"] - row["covered_in_tweet_original_rows"],
                    "covered_in_either_rows": row["covered_in_either_rows"],
                    "missing_in_either_rows": row["combined_rows"] - row["covered_in_either_rows"],
                }
            )


def main() -> None:
    print(f"Extracted CSV: {EXTRACTED_CSV}")
    print(f"Combined CSV: {COMBINED_CSV}")
    print("Building extracted text indexes...")

    extracted_en_texts, extracted_original_texts, extracted_keyword_counts, extracted_language_counts = (
        load_extracted_indexes()
    )

    total_combined_rows = 0
    total_combined_unique_tweets = 0
    seen_combined_tweets: set[str] = set()

    overall_stats: Counter = Counter()
    language_stats: dict[str, Counter] = defaultdict(Counter)
    keyword_stats: dict[str, Counter] = defaultdict(Counter)

    missing_en_rows: list[dict[str, str]] = []
    missing_original_rows: list[dict[str, str]] = []
    missing_either_rows: list[dict[str, str]] = []

    with COMBINED_CSV.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            total_combined_rows += 1

            tweet = clean_text(row.get("tweet") or "")
            tweet_norm = norm_nospace(tweet)
            if not tweet_norm:
                continue

            keyword = clean_text(row.get("keyword") or "").lower()
            language = clean_text(row.get("language") or "").lower() or "unknown"

            if tweet_norm not in seen_combined_tweets:
                seen_combined_tweets.add(tweet_norm)
                total_combined_unique_tweets += 1

            matched_in_en = tweet_norm in extracted_en_texts
            matched_in_original = tweet_norm in extracted_original_texts
            matched_in_either = matched_in_en or matched_in_original

            overall_stats["combined_rows"] += 1
            language_stats[language]["combined_rows"] += 1
            keyword_stats[keyword]["combined_rows"] += 1

            if matched_in_en:
                overall_stats["covered_in_en_text_rows"] += 1
                language_stats[language]["covered_in_en_text_rows"] += 1
                keyword_stats[keyword]["covered_in_en_text_rows"] += 1
            else:
                missing_en_rows.append(
                    {
                        "source_row": clean_text(row.get("source_row") or ""),
                        "keyword": keyword,
                        "language": language,
                        "tweet_label": clean_text(row.get("tweet_label") or ""),
                        "original_author": clean_text(row.get("original_author") or ""),
                        "tweet": tweet,
                    }
                )

            if matched_in_original:
                overall_stats["covered_in_tweet_original_rows"] += 1
                language_stats[language]["covered_in_tweet_original_rows"] += 1
                keyword_stats[keyword]["covered_in_tweet_original_rows"] += 1
            else:
                missing_original_rows.append(
                    {
                        "source_row": clean_text(row.get("source_row") or ""),
                        "keyword": keyword,
                        "language": language,
                        "tweet_label": clean_text(row.get("tweet_label") or ""),
                        "original_author": clean_text(row.get("original_author") or ""),
                        "tweet": tweet,
                    }
                )

            if matched_in_either:
                overall_stats["covered_in_either_rows"] += 1
                language_stats[language]["covered_in_either_rows"] += 1
                keyword_stats[keyword]["covered_in_either_rows"] += 1
            else:
                missing_either_rows.append(
                    {
                        "source_row": clean_text(row.get("source_row") or ""),
                        "keyword": keyword,
                        "language": language,
                        "tweet_label": clean_text(row.get("tweet_label") or ""),
                        "original_author": clean_text(row.get("original_author") or ""),
                        "tweet": tweet,
                    }
                )

            if total_combined_rows % 100000 == 0:
                print(
                    f"Scanned {total_combined_rows:,} combined rows | "
                    f"en_text covered {overall_stats['covered_in_en_text_rows']:,} | "
                    f"tweet_original covered {overall_stats['covered_in_tweet_original_rows']:,} | "
                    f"either covered {overall_stats['covered_in_either_rows']:,}"
                )

    write_missing_rows(MISSING_EN_CSV, missing_en_rows)
    write_missing_rows(MISSING_ORIGINAL_CSV, missing_original_rows)
    write_missing_rows(MISSING_EITHER_CSV, missing_either_rows)
    write_summary_by_language(SUMMARY_BY_LANGUAGE_CSV, language_stats)
    write_summary_by_keyword(SUMMARY_BY_KEYWORD_CSV, keyword_stats, extracted_keyword_counts)

    combined_rows = overall_stats["combined_rows"]
    summary_lines = [
        "EXTRACTED VS COMBINED COVERAGE",
        f"Extracted CSV: {EXTRACTED_CSV}",
        f"Combined CSV: {COMBINED_CSV}",
        "",
        "Comparison fields",
        "- combined side: `tweet`",
        "- extracted English side: `en_text`",
        "- extracted original/Hindi side: `tweet_original`",
        "",
        f"Combined rows scanned: {combined_rows:,}",
        f"Unique combined normalized tweets: {total_combined_unique_tweets:,}",
        f"Unique extracted normalized en_text values: {len(extracted_en_texts):,}",
        f"Unique extracted normalized tweet_original values: {len(extracted_original_texts):,}",
        f"Extracted language distribution: {dict(extracted_language_counts)}",
        "",
        "Overall coverage",
        (
            f"- Covered in en_text: {overall_stats['covered_in_en_text_rows']:,} / {combined_rows:,} "
            f"({pct(overall_stats['covered_in_en_text_rows'], combined_rows):.2f}%)"
        ),
        (
            f"- Missing from en_text: {combined_rows - overall_stats['covered_in_en_text_rows']:,} / {combined_rows:,} "
            f"({pct(combined_rows - overall_stats['covered_in_en_text_rows'], combined_rows):.2f}%)"
        ),
        (
            f"- Covered in tweet_original: {overall_stats['covered_in_tweet_original_rows']:,} / {combined_rows:,} "
            f"({pct(overall_stats['covered_in_tweet_original_rows'], combined_rows):.2f}%)"
        ),
        (
            f"- Missing from tweet_original: "
            f"{combined_rows - overall_stats['covered_in_tweet_original_rows']:,} / {combined_rows:,} "
            f"({pct(combined_rows - overall_stats['covered_in_tweet_original_rows'], combined_rows):.2f}%)"
        ),
        (
            f"- Covered in either field: {overall_stats['covered_in_either_rows']:,} / {combined_rows:,} "
            f"({pct(overall_stats['covered_in_either_rows'], combined_rows):.2f}%)"
        ),
        (
            f"- Missing from both fields: {combined_rows - overall_stats['covered_in_either_rows']:,} / {combined_rows:,} "
            f"({pct(combined_rows - overall_stats['covered_in_either_rows'], combined_rows):.2f}%)"
        ),
        "",
        "By combined language",
    ]

    for language in sorted(language_stats):
        row = language_stats[language]
        language_total = row["combined_rows"]
        summary_lines.extend(
            [
                f"- {language}:",
                (
                    f"  en_text covered {row['covered_in_en_text_rows']:,} / {language_total:,} "
                    f"({pct(row['covered_in_en_text_rows'], language_total):.2f}%), "
                    f"missing {language_total - row['covered_in_en_text_rows']:,}"
                ),
                (
                    f"  tweet_original covered {row['covered_in_tweet_original_rows']:,} / {language_total:,} "
                    f"({pct(row['covered_in_tweet_original_rows'], language_total):.2f}%), "
                    f"missing {language_total - row['covered_in_tweet_original_rows']:,}"
                ),
                (
                    f"  either covered {row['covered_in_either_rows']:,} / {language_total:,} "
                    f"({pct(row['covered_in_either_rows'], language_total):.2f}%), "
                    f"missing {language_total - row['covered_in_either_rows']:,}"
                ),
            ]
        )

    summary_lines.extend(
        [
            "",
            f"Missing-vs-en_text CSV: {MISSING_EN_CSV}",
            f"Missing-vs-tweet_original CSV: {MISSING_ORIGINAL_CSV}",
            f"Missing-vs-either CSV: {MISSING_EITHER_CSV}",
            f"Summary-by-language CSV: {SUMMARY_BY_LANGUAGE_CSV}",
            f"Summary-by-keyword CSV: {SUMMARY_BY_KEYWORD_CSV}",
        ]
    )

    SUMMARY_TXT.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
