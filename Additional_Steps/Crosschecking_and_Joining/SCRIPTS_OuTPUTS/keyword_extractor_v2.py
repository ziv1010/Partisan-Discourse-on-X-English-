#!/usr/bin/env python3
"""
Keyword Tweet Extractor - V2

This version is adapted for the retweet bridge dataset, which does not contain
an explicit `keyword` column. It extracts tweets by scanning `en_text` for the
configured keyword list and writes one row per matched keyword.

Matching and filtering rules
- Keyword search is performed only against `en_text`.
- Output `tweet` text is also taken from `en_text`.
- Rows are kept only when translated content meets the cosine threshold
  (`cosine_before_after >= 0.70`).
- Unchanged English rows with blank cosine are kept.
- The same tweet can appear for different keywords.
- Within a keyword, the same `tweet + author` pair is written only once.
"""

from __future__ import annotations

import csv
import html
import re
import unicodedata
from collections import Counter
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "retweet_data_combined_labeled_by_retweet_en_with_hinglish_bridge.csv"
OUT_DIR = BASE_DIR / "extracted_by_keyword"

SEARCH_TEXT_COL = "en_text"
COSINE_COL = "cosine_before_after"
MIN_COSINE = 0.70
TARGETS = ["pro ruling", "pro opposition"]


# ============================================================================
# KEYWORD LIST
# ============================================================================
KEYWORDS = [
    "ayodhya",
    "islamists",
    "balochistan",
    "sharia",
    "sangh",
    "ucc",
    "mahotsav",
    "caa",
    "aatmanirbhar",
    "unemployment",
    "inflation",
    "minorities",
    "hathras",
    "gdp",
    "msp",
    "suicides",
    "lynching",
    "spyware",
    "demonetisation",
    "democracy",
    "bhakts",
    "dictatorship",
    "ratetvdebate",
    "china",
    "congress",
    "farm laws",
    "farmers protests",
    "hindu",
    "hindutva",
    "kashmir",
    "kashmiri pandits",
    "modi",
    "muslim",
    "new parliament",
    "rahulgandhi",
    "ram mandir",
    "shaheen bagh",
]


OUTPUT_COLUMNS = [
    "source_row",
    "tweet",
    "original_author",
    "tweet_label",
    "_label_norm",
    "keyword",
    "subjects_scored",
    "stance_gold",
    "fewshot_raw",
    "fewshot_label",
    "fewshot_label_for_against",
    "fewshot_reason",
    "fewshot_shots_json",
    "language",
    "cosine_before_after",
    "row_id",
    "retweet_author",
    "retweet_party",
    "translation_route",
    "translation_engine",
    "lid_lang",
    "tweet_original",
    "en_text",
    "search_text_used",
]


NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def clean_text(value: object) -> str:
    text = html.unescape(str(value or ""))
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    return text.strip()


def norm_nospace(value: object) -> str:
    return NON_ALNUM_RE.sub("", clean_text(value).lower())


def normalize_party_tag(value: object) -> str:
    return re.sub(r"\s+", " ", clean_text(value).lower()).strip()


PRO_RULING_PARTIES = {
    "bjp",
    "jdu",
    "ljp",
    "ham",
    "jsp",
    "npp",
    "aiadmk",
    "admk",
    "ajsu",
    "agp",
    "rpi",
    "npf",
    "ipft",
    "ad (soneylal)",
    "ad soneylal",
    "ndpp",
    "rss",
    "nda",
    "mnf",
    "vhp",
    "ysrcp",
    "abvp",
    "bjd",
    "bscp",
    "jvm(p)",
    "jvmp",
    "pro ruling",
    "ruling",
}


PRO_OPPOSITION_PARTIES = {
    "inc",
    "congress",
    "rsp",
    "aitc",
    "tmc",
    "dmk",
    "sp",
    "cpim",
    "rjd",
    "gfp",
    "rlp",
    "jmm",
    "jknc",
    "iuml",
    "cpi",
    "vck",
    "cpiml",
    "cpi ml",
    "cpi(ml)",
    "mnm",
    "aap",
    "aimim",
    "aiudf",
    "asp",
    "samajwadi",
    "sdf",
    "tjs",
    "trs",
    "shiv sena",
    "sbsp",
    "tdp",
    "jds",
    "dmdk",
    "ncp",
    "rld",
    "skm",
    "ammk",
    "ijk",
    "aismk",
    "janata dalal",
    "dfyi",
    "dyfi",
    "jjp",
    "ykd",
    "aipf",
    "bva",
    "japl",
    "jcc",
    "lsp",
    "spi",
    "shp",
    "swi",
    "ldp",
    "sa",
    "jsvp",
    "ksp",
    "sfi",
    "jkync",
    "pro opp",
    "pro opposition",
    "opposition",
}


def parse_float(value: object) -> float | None:
    text = clean_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def phrase_variants(keyword: str) -> list[str]:
    parts = re.split(r"\s+or\s+|\|", clean_text(keyword), flags=re.IGNORECASE)
    variants = [part.strip().lower() for part in parts if part.strip()]
    return variants if variants else [clean_text(keyword).lower()]


def map_party_labels(raw_party: object) -> tuple[str, str]:
    party = normalize_party_tag(raw_party)
    if party in PRO_RULING_PARTIES:
        return "Pro Ruling", "pro ruling"
    if party in PRO_OPPOSITION_PARTIES:
        return "Pro OPP", "pro opposition"
    return clean_text(raw_party), "other"


def map_language(raw_lid_lang: object) -> str:
    lid = clean_text(raw_lid_lang).lower()
    if lid.startswith("eng"):
        return "english"
    if lid.startswith("hin"):
        return "hindi"
    if lid.startswith("mar"):
        return "marathi"
    if "_" in lid:
        return lid.split("_", 1)[0]
    return lid


def keyword_specs(keywords: list[str]) -> list[tuple[str, list[str]]]:
    specs = []
    for keyword in keywords:
        raw_variants = phrase_variants(keyword)
        variants = [norm_nospace(variant) for variant in raw_variants]
        variants = [variant for variant in variants if variant]
        if variants:
            specs.append((raw_variants[0], variants))
    return specs


def choose_author(row: dict[str, str]) -> str:
    for column in ("original_author", "retweet_author", "author", "user"):
        value = clean_text(row.get(column) or "")
        if value:
            return value
    return ""


def is_unchanged_english_row(row: dict[str, str], tweet_original: str, en_text: str) -> bool:
    translation_route = clean_text(row.get("translation_route") or "").lower()
    lid_lang = clean_text(row.get("lid_lang") or "").lower()

    if translation_route == "english":
        return True
    if lid_lang.startswith("eng") and norm_nospace(en_text) == norm_nospace(tweet_original):
        return True
    return False


def passes_cosine_filter(row: dict[str, str], tweet_original: str, en_text: str) -> bool:
    cosine = parse_float(row.get(COSINE_COL) or "")
    if cosine is not None:
        return cosine >= MIN_COSINE
    return is_unchanged_english_row(row, tweet_original, en_text)


def make_output_row(
    row: dict[str, str],
    keyword: str,
    source_row: str,
    en_text: str,
    tweet_label: str,
    label_norm: str,
) -> dict[str, str]:
    tweet_original = clean_text(row.get("tweet_original") or row.get("tweet") or "")

    return {
        "source_row": source_row,
        "tweet": en_text,
        "original_author": clean_text(row.get("original_author") or ""),
        "tweet_label": tweet_label,
        "_label_norm": label_norm,
        "keyword": keyword,
        "subjects_scored": "",
        "stance_gold": "",
        "fewshot_raw": "",
        "fewshot_label": "",
        "fewshot_label_for_against": "",
        "fewshot_reason": "",
        "fewshot_shots_json": "",
        "language": map_language(row.get("lid_lang") or ""),
        "cosine_before_after": clean_text(row.get(COSINE_COL) or ""),
        "row_id": clean_text(row.get("row_id") or ""),
        "retweet_author": clean_text(row.get("retweet_author") or ""),
        "retweet_party": clean_text(row.get("retweet_party") or ""),
        "translation_route": clean_text(row.get("translation_route") or ""),
        "translation_engine": clean_text(row.get("translation_engine") or ""),
        "lid_lang": clean_text(row.get("lid_lang") or ""),
        "tweet_original": tweet_original,
        "en_text": en_text,
        "search_text_used": SEARCH_TEXT_COL,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    keyword_defs = keyword_specs(KEYWORDS)
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

    print(f"CSV Path: {CSV_PATH}")
    print(f"Output Dir: {OUT_DIR}")
    print(f"Search column: {SEARCH_TEXT_COL}")
    print(f"Cosine filter: translated rows with {COSINE_COL} >= {MIN_COSINE:.2f}")
    print(f"Total keywords: {len(KEYWORDS)}")
    print("Mode: EN_TEXT SEARCH ONLY")
    print("Dedup Rule: Same en_text + same author NOT allowed within same keyword")

    per_keyword_csv_handles = {}
    per_keyword_id_handles = {}
    per_keyword_writers = {}

    combined_csv = OUT_DIR / f"extracted_ALL_{len(KEYWORDS)}keywords_cosine070.csv"
    combined_ids = OUT_DIR / f"extracted_ALL_{len(KEYWORDS)}keywords_cosine070_ids.txt"
    summary_csv = OUT_DIR / "extraction_summary.csv"

    try:
        for keyword, _variants in keyword_defs:
            canonical_name = keyword.replace(" ", "_")
            csv_path = OUT_DIR / f"extracted_{canonical_name}.csv"
            ids_path = OUT_DIR / f"extracted_{canonical_name}_ids.txt"
            csv_handle = csv_path.open("w", newline="", encoding="utf-8")
            id_handle = ids_path.open("w", encoding="utf-8")
            writer = csv.DictWriter(csv_handle, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()

            per_keyword_csv_handles[keyword] = csv_handle
            per_keyword_id_handles[keyword] = id_handle
            per_keyword_writers[keyword] = writer

        with CSV_PATH.open(newline="", encoding="utf-8") as infile, \
            combined_csv.open("w", newline="", encoding="utf-8") as combined_out, \
            combined_ids.open("w", encoding="utf-8") as combined_ids_out:

            reader = csv.DictReader(infile)
            if reader.fieldnames is None:
                raise ValueError("Input CSV has no header row.")
            if SEARCH_TEXT_COL not in reader.fieldnames:
                raise ValueError(
                    f"Required column '{SEARCH_TEXT_COL}' not found. Available columns: {reader.fieldnames}"
                )

            combined_writer = csv.DictWriter(combined_out, fieldnames=OUTPUT_COLUMNS)
            combined_writer.writeheader()

            for row_index, row in enumerate(reader):
                total_rows_scanned += 1

                source_row = clean_text(row.get("row_id") or row.get("source_row") or row_index)
                tweet_original = clean_text(row.get("tweet_original") or row.get("tweet") or "")
                en_text = clean_text(row.get(SEARCH_TEXT_COL) or "")
                if not en_text:
                    continue

                tweet_label, label_norm = map_party_labels(row.get("retweet_party") or "")
                if label_norm not in TARGETS:
                    continue

                cosine = parse_float(row.get(COSINE_COL) or "")
                if cosine is None:
                    if is_unchanged_english_row(row, tweet_original, en_text):
                        translation_stats["blank_cosine_kept_as_english"] += 1
                    else:
                        translation_stats["blank_cosine_filtered"] += 1
                        continue
                elif cosine < MIN_COSINE:
                    translation_stats["low_cosine_filtered"] += 1
                    continue
                else:
                    translation_stats["cosine_kept"] += 1

                search_text_norm = norm_nospace(en_text)
                if not search_text_norm:
                    continue

                author = choose_author(row)
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
                    output_row = make_output_row(
                        row=row,
                        keyword=keyword,
                        source_row=source_row,
                        en_text=en_text,
                        tweet_label=tweet_label,
                        label_norm=label_norm,
                    )

                    per_keyword_writers[keyword].writerow(output_row)
                    per_keyword_id_handles[keyword].write(f"{source_row}\n")
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

        with summary_csv.open("w", newline="", encoding="utf-8") as summary_out:
            fieldnames = [
                "keyword",
                "total",
                "pro_ruling",
                "pro_opposition",
            ]
            writer = csv.DictWriter(summary_out, fieldnames=fieldnames)
            writer.writeheader()

            print("\n" + "=" * 80)
            print("SUMMARY")
            print("=" * 80)
            print(f"Rows scanned: {total_rows_scanned:,}")
            print(f"Rows written: {total_rows_written:,}")
            print(f"Cosine filter: {COSINE_COL} >= {MIN_COSINE:.2f} for translated rows")
            print(f"Blank cosine kept as unchanged English rows: {translation_stats['blank_cosine_kept_as_english']:,}")
            print(f"Blank cosine filtered: {translation_stats['blank_cosine_filtered']:,}")
            print(f"Low cosine filtered: {translation_stats['low_cosine_filtered']:,}")
            print(f"Rows kept with numeric cosine >= threshold: {translation_stats['cosine_kept']:,}")
            print("")

            for keyword, _variants in keyword_defs:
                row = {
                    "keyword": keyword,
                    "total": stats[keyword]["total_extracted"],
                    "pro_ruling": stats[keyword]["by_label"].get("pro ruling", 0),
                    "pro_opposition": stats[keyword]["by_label"].get("pro opposition", 0),
                }
                writer.writerow(row)
                print(
                    f"{keyword:20} total={row['total']:6} | "
                    f"pro_ruling={row['pro_ruling']:6} | "
                    f"pro_opposition={row['pro_opposition']:6}"
                )

        print("\nFiles saved:")
        print(f"- Combined CSV: {combined_csv}")
        print(f"- Combined IDs: {combined_ids}")
        print(f"- Summary CSV: {summary_csv}")

    finally:
        for handle in per_keyword_csv_handles.values():
            handle.close()
        for handle in per_keyword_id_handles.values():
            handle.close()


if __name__ == "__main__":
    main()
