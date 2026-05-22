#!/usr/bin/env python3
"""
Classify why rows from `combined_stance_results.csv` are missing from
`extracted_ALL_37keywords_cosine070.csv`, using exact normalized tweet-text
 matches against the bridge CSV rather than source_row / row_id.

Outputs:
- missing_reason_by_text_classification.csv
- missing_reason_by_text_summary.txt
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
EXTRACTOR_SCRIPT = BASE_DIR / "keyword_extractor_v2.py"

OUTPUT_CSV = BASE_DIR / "missing_reason_by_text_classification.csv"
OUTPUT_SUMMARY = BASE_DIR / "missing_reason_by_text_summary.txt"

TEXT_FIELDS = [
    "tweet",
    "en_text",
    "tweet_original",
    "hinglish_english_text",
    "hinglish_hindi_text",
    "hinglish_roman_text",
]
EXTRACTION_FIELDS = {"en_text", "tweet_original"}


def load_extractor_module():
    spec = importlib.util.spec_from_file_location("keyword_extractor_v2", EXTRACTOR_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_missing_rows(extractor):
    rows = []
    norms = set()
    with MISSING_CSV.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            tweet = extractor.clean_text(row.get("tweet") or "")
            tweet_norm = extractor.norm_nospace(tweet)
            if not tweet_norm:
                continue
            item = {
                "source_row": extractor.clean_text(row.get("source_row") or ""),
                "keyword": extractor.clean_text(row.get("keyword") or "").lower(),
                "language": extractor.clean_text(row.get("language") or "").lower() or "unknown",
                "tweet_label": extractor.clean_text(row.get("tweet_label") or ""),
                "original_author": extractor.clean_text(row.get("original_author") or ""),
                "tweet": tweet,
                "tweet_norm": tweet_norm,
            }
            rows.append(item)
            norms.add(tweet_norm)
    return rows, norms


def stage_label(stage: int) -> str:
    return {
        0: "present_in_extraction_field",
        1: "en_text_available",
        2: "label_ok",
        3: "cosine_ok",
        4: "keyword_match",
    }.get(stage, "")


def main() -> None:
    extractor = load_extractor_module()
    keyword_defs = extractor.keyword_specs(extractor.KEYWORDS)
    missing_rows, missing_norms = load_missing_rows(extractor)

    norm_info: dict[str, dict[str, object]] = {
        norm: {
            "matched_fields": set(),
            "matched_extraction_fields": set(),
            "best_stage": -1,
            "best_stage_label": "",
            "best_row_id": "",
            "best_retweet_party": "",
            "best_cosine": "",
            "best_translation_route": "",
            "best_matching_keywords": "",
            "best_en_text": "",
            "best_tweet_original": "",
        }
        for norm in missing_norms
    }

    with BRIDGE_CSV.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for rows_scanned, row in enumerate(reader, 1):
            row_id = extractor.clean_text(row.get("row_id") or "")
            en_text = extractor.clean_text(row.get("en_text") or "")
            en_text_norm = extractor.norm_nospace(en_text)
            tweet_original = extractor.clean_text(row.get("tweet_original") or row.get("tweet") or "")

            tweet_label, label_norm = extractor.map_party_labels(row.get("retweet_party") or "")
            label_ok = label_norm in extractor.TARGETS
            cosine_ok = extractor.passes_cosine_filter(row, tweet_original, en_text)
            matching_keywords = []
            if en_text_norm:
                for keyword, variants in keyword_defs:
                    if any(variant in en_text_norm for variant in variants):
                        matching_keywords.append(keyword)

            for field in TEXT_FIELDS:
                field_norm = extractor.norm_nospace(row.get(field) or "")
                if not field_norm or field_norm not in norm_info:
                    continue

                info = norm_info[field_norm]
                info["matched_fields"].add(field)

                stage = -1
                if field in EXTRACTION_FIELDS:
                    info["matched_extraction_fields"].add(field)
                    stage = 0
                    if en_text_norm:
                        stage = 1
                        if label_ok:
                            stage = 2
                            if cosine_ok:
                                stage = 3
                                if matching_keywords:
                                    stage = 4

                if stage > info["best_stage"]:
                    info["best_stage"] = stage
                    info["best_stage_label"] = stage_label(stage)
                    info["best_row_id"] = row_id
                    info["best_retweet_party"] = extractor.clean_text(row.get("retweet_party") or "")
                    info["best_cosine"] = extractor.clean_text(row.get("cosine_before_after") or "")
                    info["best_translation_route"] = extractor.clean_text(row.get("translation_route") or "")
                    info["best_matching_keywords"] = "|".join(matching_keywords)
                    info["best_en_text"] = en_text
                    info["best_tweet_original"] = tweet_original

            if rows_scanned % 1_000_000 == 0:
                matched = sum(1 for info in norm_info.values() if info["matched_fields"])
                print(f"Scanned {rows_scanned:,} bridge rows | matched missing texts {matched:,} / {len(norm_info):,}")

    results = []
    reason_counts = Counter()
    reason_by_language: dict[str, Counter] = defaultdict(Counter)
    reason_by_keyword: dict[str, Counter] = defaultdict(Counter)

    for row in missing_rows:
        info = norm_info[row["tweet_norm"]]
        matched_fields = sorted(info["matched_fields"])
        matched_extraction_fields = sorted(info["matched_extraction_fields"])
        best_stage = info["best_stage"]

        if not matched_fields:
            reason = "not_present_in_bridge_text_fields"
        elif not matched_extraction_fields:
            reason = "present_only_outside_extraction_fields"
        elif best_stage == 0:
            reason = "missing_en_text"
        elif best_stage == 1:
            reason = "filtered_non_target_label"
        elif best_stage == 2:
            reason = "filtered_cosine"
        elif best_stage == 3:
            reason = "no_keyword_match_in_en_text"
        elif best_stage == 4:
            reason = "extractable_but_missing_from_extracted"
        else:
            reason = "unclassified"

        result = {
            "source_row": row["source_row"],
            "keyword": row["keyword"],
            "language": row["language"],
            "tweet_label": row["tweet_label"],
            "original_author": row["original_author"],
            "tweet": row["tweet"],
            "reason": reason,
            "matched_fields": "|".join(matched_fields),
            "matched_extraction_fields": "|".join(matched_extraction_fields),
            "best_stage_label": info["best_stage_label"],
            "best_row_id": info["best_row_id"],
            "best_retweet_party": info["best_retweet_party"],
            "best_cosine": info["best_cosine"],
            "best_translation_route": info["best_translation_route"],
            "best_matching_keywords": info["best_matching_keywords"],
            "best_en_text": info["best_en_text"],
            "best_tweet_original": info["best_tweet_original"],
        }
        results.append(result)

        reason_counts[reason] += 1
        reason_by_language[row["language"]][reason] += 1
        reason_by_keyword[row["keyword"]][reason] += 1

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as outfile:
        fieldnames = [
            "source_row",
            "keyword",
            "language",
            "tweet_label",
            "original_author",
            "tweet",
            "reason",
            "matched_fields",
            "matched_extraction_fields",
            "best_stage_label",
            "best_row_id",
            "best_retweet_party",
            "best_cosine",
            "best_translation_route",
            "best_matching_keywords",
            "best_en_text",
            "best_tweet_original",
        ]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    summary_lines = [
        "MISSING REASON CLASSIFICATION BY TEXT",
        f"Rows classified: {len(results):,}",
        "",
        "Overall reason counts",
    ]
    for reason, count in reason_counts.most_common():
        summary_lines.append(f"- {reason}: {count:,}")

    summary_lines.extend(["", "By language"])
    for language in sorted(reason_by_language):
        summary_lines.append(f"- {language}: {dict(reason_by_language[language])}")

    summary_lines.extend(["", f"Detailed CSV: {OUTPUT_CSV}"])
    OUTPUT_SUMMARY.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
