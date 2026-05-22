#!/usr/bin/env python3
"""
Classify why rows from `combined_stance_results.csv` are missing from
`extracted_ALL_37keywords_cosine070.csv`.

The script uses:
- coverage_missing_vs_either.csv as the set of rows missing from extracted
- the bridge CSV as the source of truth for row attributes
- the keyword extractor logic from keyword_extractor_v2.py

Outputs:
- missing_reason_classification.csv
- missing_reason_summary.txt
"""

from __future__ import annotations

import csv
import importlib.util
from collections import Counter, defaultdict
from pathlib import Path


BASE_DIR = Path(
    "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Additional_Steps/Crosschecking_and_Joining"
)
MISSING_CSV = BASE_DIR / "coverage_missing_vs_either.csv"
BRIDGE_CSV = BASE_DIR / "retweet_data_combined_labeled_by_retweet_en_with_hinglish_bridge.csv"
EXTRACTED_CSV = BASE_DIR / "extracted_by_keyword" / "extracted_ALL_37keywords_cosine070.csv"
DUPLICATE_MAP_CSV = BASE_DIR / "english_missing_not_in_bridge_by_id_but_found_by_text.csv"
EXTRACTOR_SCRIPT = BASE_DIR / "keyword_extractor_v2.py"

OUTPUT_REASON_CSV = BASE_DIR / "missing_reason_classification.csv"
OUTPUT_SUMMARY_TXT = BASE_DIR / "missing_reason_summary.txt"


def load_extractor_module():
    spec = importlib.util.spec_from_file_location("keyword_extractor_v2", EXTRACTOR_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_extracted_texts(extractor) -> set[str]:
    extracted_texts: set[str] = set()
    with EXTRACTED_CSV.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            for field in ("en_text", "tweet_original", "tweet"):
                text_norm = extractor.norm_nospace(row.get(field) or "")
                if text_norm:
                    extracted_texts.add(text_norm)
    return extracted_texts


def load_duplicate_map() -> dict[str, str]:
    duplicate_map: dict[str, str] = {}
    if not DUPLICATE_MAP_CSV.exists():
        return duplicate_map

    with DUPLICATE_MAP_CSV.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            source_row = (row.get("source_row") or "").strip()
            sample_ids = (row.get("sample_bridge_row_ids") or "").strip()
            if not source_row or not sample_ids:
                continue
            first_sample_id = sample_ids.split("|", 1)[0].strip()
            if first_sample_id:
                duplicate_map[source_row] = first_sample_id
    return duplicate_map


def load_missing_rows(extractor) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with MISSING_CSV.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            tweet = extractor.clean_text(row.get("tweet") or "")
            rows.append(
                {
                    "source_row": extractor.clean_text(row.get("source_row") or ""),
                    "keyword": extractor.clean_text(row.get("keyword") or "").lower(),
                    "language": extractor.clean_text(row.get("language") or "").lower() or "unknown",
                    "tweet_label": extractor.clean_text(row.get("tweet_label") or ""),
                    "original_author": extractor.clean_text(row.get("original_author") or ""),
                    "tweet": tweet,
                    "tweet_norm": extractor.norm_nospace(tweet),
                }
            )
    return rows


def load_bridge_rows(candidate_ids: set[str]) -> dict[str, dict[str, str]]:
    bridge_rows: dict[str, dict[str, str]] = {}
    with BRIDGE_CSV.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for rows_scanned, row in enumerate(reader, 1):
            row_id = (row.get("row_id") or "").strip()
            if row_id in candidate_ids:
                bridge_rows[row_id] = row
                if len(bridge_rows) == len(candidate_ids):
                    break
            if rows_scanned % 1_000_000 == 0:
                print(f"Scanned {rows_scanned:,} bridge rows | captured {len(bridge_rows):,} / {len(candidate_ids):,}")
    return bridge_rows


def classify_row(extractor, keyword_defs, extracted_texts: set[str], missing_row: dict[str, str], bridge_row: dict[str, str]):
    en_text = extractor.clean_text(bridge_row.get("en_text") or "")
    en_text_norm = extractor.norm_nospace(en_text)

    tweet_original = extractor.clean_text(bridge_row.get("tweet_original") or bridge_row.get("tweet") or "")
    tweet_original_norm = extractor.norm_nospace(tweet_original)

    tweet_label, label_norm = extractor.map_party_labels(bridge_row.get("retweet_party") or "")
    label_ok = label_norm in extractor.TARGETS
    cosine_ok = extractor.passes_cosine_filter(bridge_row, tweet_original, en_text)

    matching_keywords = []
    if en_text_norm:
        for keyword, variants in keyword_defs:
            if any(variant in en_text_norm for variant in variants):
                matching_keywords.append(keyword)

    any_keyword_match = bool(matching_keywords)
    combined_keyword_match = missing_row["keyword"] in matching_keywords

    combined_tweet_matches_en = missing_row["tweet_norm"] == en_text_norm if missing_row["tweet_norm"] and en_text_norm else False
    combined_tweet_matches_original = (
        missing_row["tweet_norm"] == tweet_original_norm if missing_row["tweet_norm"] and tweet_original_norm else False
    )

    candidate_text_present_in_extracted = False
    for text_norm in (en_text_norm, tweet_original_norm):
        if text_norm and text_norm in extracted_texts:
            candidate_text_present_in_extracted = True
            break

    if not label_ok:
        reason = "filtered_non_target_label"
    elif not cosine_ok:
        reason = "filtered_cosine"
    elif not en_text_norm:
        reason = "missing_en_text"
    elif not any_keyword_match:
        reason = "no_keyword_match_in_en_text"
    elif candidate_text_present_in_extracted and not (combined_tweet_matches_en or combined_tweet_matches_original):
        reason = "text_mismatch_between_combined_and_bridge"
    elif candidate_text_present_in_extracted and (combined_tweet_matches_en or combined_tweet_matches_original):
        reason = "unexpected_coverage_miss"
    else:
        reason = "extractable_but_missing_from_extracted"

    return {
        "reason": reason,
        "candidate_label_norm": label_norm,
        "candidate_label_ok": "yes" if label_ok else "no",
        "candidate_cosine_ok": "yes" if cosine_ok else "no",
        "candidate_en_text_nonempty": "yes" if en_text_norm else "no",
        "candidate_any_keyword_match": "yes" if any_keyword_match else "no",
        "candidate_combined_keyword_match": "yes" if combined_keyword_match else "no",
        "candidate_matching_keywords": "|".join(matching_keywords),
        "combined_tweet_matches_bridge_en_text": "yes" if combined_tweet_matches_en else "no",
        "combined_tweet_matches_bridge_tweet_original": "yes" if combined_tweet_matches_original else "no",
        "candidate_text_present_in_extracted": "yes" if candidate_text_present_in_extracted else "no",
        "bridge_en_text": en_text,
        "bridge_tweet_original": tweet_original,
        "bridge_retweet_party": extractor.clean_text(bridge_row.get("retweet_party") or ""),
        "bridge_cosine_before_after": extractor.clean_text(bridge_row.get("cosine_before_after") or ""),
        "bridge_translation_route": extractor.clean_text(bridge_row.get("translation_route") or ""),
    }


def main() -> None:
    extractor = load_extractor_module()
    keyword_defs = extractor.keyword_specs(extractor.KEYWORDS)
    extracted_texts = load_extracted_texts(extractor)
    duplicate_map = load_duplicate_map()
    missing_rows = load_missing_rows(extractor)

    candidate_ids: set[str] = set()
    for row in missing_rows:
        if row["source_row"]:
            candidate_ids.add(row["source_row"])
        duplicate_id = duplicate_map.get(row["source_row"])
        if duplicate_id:
            candidate_ids.add(duplicate_id)

    bridge_rows = load_bridge_rows(candidate_ids)

    reason_counts = Counter()
    reason_by_language: dict[str, Counter] = defaultdict(Counter)
    reason_by_keyword: dict[str, Counter] = defaultdict(Counter)
    combined_keyword_no_match_counts = Counter()
    results = []

    for row in missing_rows:
        candidate_row = None
        candidate_lookup = ""

        if row["source_row"] and row["source_row"] in bridge_rows:
            candidate_row = bridge_rows[row["source_row"]]
            candidate_lookup = "source_row"
        else:
            duplicate_id = duplicate_map.get(row["source_row"], "")
            if duplicate_id and duplicate_id in bridge_rows:
                candidate_row = bridge_rows[duplicate_id]
                candidate_lookup = "text_duplicate_row_id"

        if candidate_row is None:
            reason = "no_bridge_candidate"
            result = {
                **row,
                "candidate_lookup": "",
                "candidate_row_id": "",
                "reason": reason,
                "candidate_label_norm": "",
                "candidate_label_ok": "no",
                "candidate_cosine_ok": "no",
                "candidate_en_text_nonempty": "no",
                "candidate_any_keyword_match": "no",
                "candidate_combined_keyword_match": "no",
                "candidate_matching_keywords": "",
                "combined_tweet_matches_bridge_en_text": "no",
                "combined_tweet_matches_bridge_tweet_original": "no",
                "candidate_text_present_in_extracted": "no",
                "bridge_en_text": "",
                "bridge_tweet_original": "",
                "bridge_retweet_party": "",
                "bridge_cosine_before_after": "",
                "bridge_translation_route": "",
            }
        else:
            classified = classify_row(extractor, keyword_defs, extracted_texts, row, candidate_row)
            reason = classified["reason"]
            if classified["candidate_combined_keyword_match"] == "no":
                combined_keyword_no_match_counts[row["keyword"]] += 1
            result = {
                **row,
                "candidate_lookup": candidate_lookup,
                "candidate_row_id": extractor.clean_text(candidate_row.get("row_id") or ""),
                **classified,
            }

            results.append(result)
        reason_counts[reason] += 1
        reason_by_language[row["language"]][reason] += 1
        reason_by_keyword[row["keyword"]][reason] += 1

    with OUTPUT_REASON_CSV.open("w", newline="", encoding="utf-8") as outfile:
        fieldnames = [
            "source_row",
            "keyword",
            "language",
            "tweet_label",
            "original_author",
            "tweet",
            "candidate_lookup",
            "candidate_row_id",
            "reason",
            "candidate_label_norm",
            "candidate_label_ok",
            "candidate_cosine_ok",
            "candidate_en_text_nonempty",
            "candidate_any_keyword_match",
            "candidate_combined_keyword_match",
            "candidate_matching_keywords",
            "combined_tweet_matches_bridge_en_text",
            "combined_tweet_matches_bridge_tweet_original",
            "candidate_text_present_in_extracted",
            "bridge_retweet_party",
            "bridge_cosine_before_after",
            "bridge_translation_route",
            "bridge_en_text",
            "bridge_tweet_original",
        ]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row_out = {key: value for key, value in result.items() if key in fieldnames}
            writer.writerow(row_out)

    summary_lines = [
        "MISSING REASON CLASSIFICATION",
        f"Rows classified: {len(results):,}",
        "",
        "Overall reason counts",
    ]
    for reason, count in reason_counts.most_common():
        summary_lines.append(f"- {reason}: {count:,}")

    summary_lines.extend(["", "By language"])
    for language in sorted(reason_by_language):
        summary_lines.append(f"- {language}: {dict(reason_by_language[language])}")

    summary_lines.extend(["", "Top keywords where combined keyword is not found in bridge en_text"])
    for keyword, count in combined_keyword_no_match_counts.most_common(20):
        summary_lines.append(f"- {keyword}: {count:,}")

    summary_lines.extend(["", f"Detailed CSV: {OUTPUT_REASON_CSV}"])
    OUTPUT_SUMMARY_TXT.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
