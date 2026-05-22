#!/usr/bin/env python3
"""
Extract keyword-matched candidate tweets from the updated retweet dataset and
compare them against the existing combined stance dataset.

Matching approach
- Keyword list comes from the existing combined stance CSV.
- Keyword detection uses English search text when available (`en_text`),
  otherwise falls back to the original tweet text.
- Overlap is measured on (keyword, original tweet text), normalized to be
  case-insensitive and robust to punctuation / spacing differences.

Outputs
- retweet_keyword_candidates_expanded.csv
- retweet_keyword_candidates_remaining_to_add.csv
- retweet_keyword_candidates_overlap_summary.csv
- retweet_keyword_candidates_run_summary.txt
"""

from __future__ import annotations

import argparse
import csv
import html
import re
import unicodedata
from collections import Counter
from pathlib import Path


BASE_DIR = Path(
    "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Additional_Steps/Crosschecking_and_Joining"
)
EXISTING_CSV = BASE_DIR / "combined_stance_results.csv"
UPDATED_CSV = BASE_DIR / "retweet_data_combined_labeled_by_retweet_en.csv"

DEFAULT_TRANSLATION_COSINE_THRESHOLD = 0.77


STANDARD_OUTPUT_COLUMNS = [
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
]

EXTRA_OUTPUT_COLUMNS = [
    "row_id",
    "retweet_author",
    "retweet_party",
    "lid_lang",
    "script",
    "translation_route",
    "translation_engine",
    "search_text_used",
    "tweet_original",
    "en_text",
]

OUTPUT_COLUMNS = STANDARD_OUTPUT_COLUMNS + EXTRA_OUTPUT_COLUMNS


NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def clean_text(value: str) -> str:
    text = html.unescape(str(value or ""))
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    return text.strip()


def norm_nospace(value: str) -> str:
    return NON_ALNUM_RE.sub("", clean_text(value).lower())


def normalize_party_tag(value: str) -> str:
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


def map_party_labels(raw_party: str) -> tuple[str, str]:
    party = normalize_party_tag(raw_party)
    if party in PRO_RULING_PARTIES:
        return "Pro Ruling", "pro ruling"
    if party in PRO_OPPOSITION_PARTIES:
        return "Pro OPP", "pro opposition"
    return clean_text(raw_party), ""


def map_language(raw_lid_lang: str) -> str:
    lid = clean_text(raw_lid_lang).lower()
    if lid.startswith("eng"):
        return "english"
    if lid.startswith("hin"):
        return "hindi"
    if lid.startswith("mar"):
        return "marathi"
    if "_" in lid:
        return lid.split("_", 1)[0]
    return lid or ""


def parse_float(value: str) -> float | None:
    text = clean_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def threshold_suffix(threshold: float) -> str:
    compact = str(threshold).replace(".", "")
    return f"cosine{compact}"


def is_translated_row(row: dict[str, str], original_tweet: str, en_text: str) -> bool:
    lid = clean_text(row.get("lid_lang") or "").lower()
    if lid and not lid.startswith("eng"):
        return True
    if en_text and norm_nospace(en_text) != norm_nospace(original_tweet):
        return True
    return False


def load_existing_pairs() -> tuple[list[str], set[tuple[str, str]], Counter]:
    keyword_counts: Counter[str] = Counter()
    existing_pairs: set[tuple[str, str]] = set()

    with EXISTING_CSV.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            keyword = clean_text(row.get("keyword", "")).lower()
            tweet = row.get("tweet") or ""
            tweet_norm = norm_nospace(tweet)

            if not keyword or not tweet_norm:
                continue

            keyword_counts[keyword] += 1
            existing_pairs.add((keyword, tweet_norm))

    return sorted(keyword_counts), existing_pairs, keyword_counts


def build_keyword_specs(keywords: list[str]) -> list[tuple[str, str]]:
    specs = []
    for keyword in keywords:
        keyword_norm = norm_nospace(keyword)
        if keyword_norm:
            specs.append((keyword, keyword_norm))
    # Longer compact keywords first is slightly safer for human inspection.
    specs.sort(key=lambda item: (-len(item[1]), item[0]))
    return specs


def make_output_row(row: dict[str, str], keyword: str, search_text_used: str) -> dict[str, str]:
    tweet_original = clean_text(row.get("tweet_original") or row.get("tweet") or "")
    tweet_label, label_norm = map_party_labels(row.get("retweet_party", ""))

    return {
        "source_row": clean_text(row.get("row_id") or row.get("source_row") or ""),
        "tweet": tweet_original,
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
        "language": map_language(row.get("lid_lang", "")),
        "cosine_before_after": clean_text(row.get("cosine_before_after") or ""),
        "row_id": clean_text(row.get("row_id") or ""),
        "retweet_author": clean_text(row.get("retweet_author") or ""),
        "retweet_party": clean_text(row.get("retweet_party") or ""),
        "lid_lang": clean_text(row.get("lid_lang") or ""),
        "script": clean_text(row.get("script") or ""),
        "translation_route": clean_text(row.get("translation_route") or ""),
        "translation_engine": clean_text(row.get("translation_engine") or ""),
        "search_text_used": search_text_used,
        "tweet_original": tweet_original,
        "en_text": clean_text(row.get("en_text") or ""),
    }


def write_summary(
    keywords: list[str],
    existing_keyword_counts: Counter,
    summary_stats: dict[str, Counter],
    raw_party_counts: Counter,
    raw_language_counts: Counter,
    total_rows_scanned: int,
    translation_stats: Counter,
    threshold: float,
    output_candidates: Path,
    output_remaining: Path,
    output_summary_csv: Path,
) -> None:
    with output_summary_csv.open("w", newline="", encoding="utf-8") as outfile:
        fieldnames = [
            "keyword",
            "existing_rows_in_combined",
            "raw_keyword_hits_in_updated",
            "unique_candidate_keyword_tweet_pairs",
            "already_in_combined",
            "remaining_to_add",
        ]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for keyword in keywords:
            writer.writerow(
                {
                    "keyword": keyword,
                    "existing_rows_in_combined": existing_keyword_counts.get(keyword, 0),
                    "raw_keyword_hits_in_updated": summary_stats["raw_hits"].get(keyword, 0),
                    "unique_candidate_keyword_tweet_pairs": summary_stats["unique_candidates"].get(keyword, 0),
                    "already_in_combined": summary_stats["overlap"].get(keyword, 0),
                    "remaining_to_add": summary_stats["remaining"].get(keyword, 0),
                }
            )

    total_raw_hits = sum(summary_stats["raw_hits"].values())
    total_unique_candidates = sum(summary_stats["unique_candidates"].values())
    total_overlap = sum(summary_stats["overlap"].values())
    total_remaining = sum(summary_stats["remaining"].values())

    lines = [
        "UPDATED KEYWORD CANDIDATE EXTRACTION",
        f"Existing combined CSV: {EXISTING_CSV}",
        f"Updated retweet CSV: {UPDATED_CSV}",
        f"Rows scanned in updated CSV: {total_rows_scanned:,}",
        f"Keywords used: {len(keywords)}",
        f"Translation cosine threshold: {threshold:.2f}",
        "",
        "Overall counts",
        f"- Raw keyword hits in updated CSV: {total_raw_hits:,}",
        f"- Unique candidate keyword+tweets: {total_unique_candidates:,}",
        f"- Already present in combined: {total_overlap:,}",
        f"- Remaining to add: {total_remaining:,}",
        "",
        "Translation filtering",
        f"- Translated rows seen: {translation_stats['translated_seen']:,}",
        f"- Translated rows filtered for cosine < {threshold:.2f}: {translation_stats['translated_filtered_low_cosine']:,}",
        f"- Translated rows kept with missing cosine: {translation_stats['translated_missing_cosine_kept']:,}",
        f"- Translated rows kept after filter: {translation_stats['translated_kept']:,}",
        "",
        "Updated CSV distribution",
        f"- retweet_party counts: {dict(raw_party_counts)}",
        f"- lid_lang->language counts: {dict(raw_language_counts)}",
        "",
        f"Candidate CSV: {output_candidates}",
        f"Remaining-only CSV: {output_remaining}",
        f"Per-keyword summary CSV: {output_summary_csv}",
    ]

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_TRANSLATION_COSINE_THRESHOLD,
        help="Minimum cosine_before_after for translated rows to be kept.",
    )
    args = parser.parse_args()

    suffix = threshold_suffix(args.threshold)
    output_candidates = BASE_DIR / f"retweet_keyword_candidates_expanded_{suffix}.csv"
    output_remaining = BASE_DIR / f"retweet_keyword_candidates_remaining_to_add_{suffix}.csv"
    output_summary_csv = BASE_DIR / f"retweet_keyword_candidates_overlap_summary_{suffix}.csv"
    output_summary_txt = BASE_DIR / f"retweet_keyword_candidates_run_summary_{suffix}.txt"

    keywords, existing_pairs, existing_keyword_counts = load_existing_pairs()
    keyword_specs = build_keyword_specs(keywords)

    summary_stats: dict[str, Counter] = {
        "raw_hits": Counter(),
        "unique_candidates": Counter(),
        "overlap": Counter(),
        "remaining": Counter(),
    }

    raw_party_counts: Counter[str] = Counter()
    raw_language_counts: Counter[str] = Counter()
    translation_stats: Counter[str] = Counter()

    seen_candidate_pairs: set[tuple[str, str]] = set()
    total_rows_scanned = 0

    with UPDATED_CSV.open(newline="", encoding="utf-8") as infile, \
        output_candidates.open("w", newline="", encoding="utf-8") as candidates_out, \
        output_remaining.open("w", newline="", encoding="utf-8") as remaining_out:

        reader = csv.DictReader(infile)
        candidate_writer = csv.DictWriter(candidates_out, fieldnames=OUTPUT_COLUMNS)
        remaining_writer = csv.DictWriter(remaining_out, fieldnames=OUTPUT_COLUMNS)
        candidate_writer.writeheader()
        remaining_writer.writeheader()

        for row in reader:
            total_rows_scanned += 1

            raw_party_counts[clean_text(row.get("retweet_party") or "")] += 1
            raw_language_counts[map_language(row.get("lid_lang") or "")] += 1

            original_tweet = clean_text(row.get("tweet_original") or row.get("tweet") or "")
            original_tweet_norm = norm_nospace(original_tweet)
            if not original_tweet_norm:
                continue

            en_text = clean_text(row.get("en_text") or "")
            translated = is_translated_row(row, original_tweet, en_text)
            cosine = parse_float(row.get("cosine_before_after") or "")

            if translated:
                translation_stats["translated_seen"] += 1
                if cosine is None:
                    translation_stats["translated_missing_cosine_kept"] += 1
                elif cosine < args.threshold:
                    translation_stats["translated_filtered_low_cosine"] += 1
                    continue
                translation_stats["translated_kept"] += 1

            search_text = en_text or original_tweet
            search_text_norm = norm_nospace(search_text)
            if not search_text_norm:
                continue

            matched_keywords = []
            for keyword, keyword_norm in keyword_specs:
                if keyword_norm in search_text_norm:
                    matched_keywords.append(keyword)
                    summary_stats["raw_hits"][keyword] += 1

            if not matched_keywords:
                continue

            for keyword in matched_keywords:
                pair = (keyword, original_tweet_norm)
                if pair in seen_candidate_pairs:
                    continue

                seen_candidate_pairs.add(pair)
                summary_stats["unique_candidates"][keyword] += 1

                output_row = make_output_row(row, keyword, "en_text" if clean_text(row.get("en_text") or "") else "tweet")
                candidate_writer.writerow(output_row)

                if pair in existing_pairs:
                    summary_stats["overlap"][keyword] += 1
                else:
                    summary_stats["remaining"][keyword] += 1
                    remaining_writer.writerow(output_row)

            if total_rows_scanned % 100000 == 0:
                print(
                    f"Scanned {total_rows_scanned:,} rows | "
                    f"unique candidates {sum(summary_stats['unique_candidates'].values()):,} | "
                    f"remaining {sum(summary_stats['remaining'].values()):,}"
                )

    summary_text = write_summary(
        keywords=keywords,
        existing_keyword_counts=existing_keyword_counts,
        summary_stats=summary_stats,
        raw_party_counts=raw_party_counts,
        raw_language_counts=raw_language_counts,
        total_rows_scanned=total_rows_scanned,
        translation_stats=translation_stats,
        threshold=args.threshold,
        output_candidates=output_candidates,
        output_remaining=output_remaining,
        output_summary_csv=output_summary_csv,
    )
    output_summary_txt.write_text(summary_text, encoding="utf-8")

    print("Done.")
    print(f"Candidate CSV: {output_candidates}")
    print(f"Remaining CSV: {output_remaining}")
    print(f"Summary CSV: {output_summary_csv}")
    print(f"Summary TXT: {output_summary_txt}")


if __name__ == "__main__":
    main()
