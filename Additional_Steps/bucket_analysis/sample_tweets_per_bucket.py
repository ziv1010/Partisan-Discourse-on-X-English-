#!/usr/bin/env python3
"""
Sample tweet-level examples for auditing thematic bucket assignments.

The bucket assignment used in the final visualisation code is keyword based:
each tweet inherits the thematic bucket assigned to its keyword/aspect. This
script applies the same mapping to combined_stance_results.csv and writes a
deterministic review sample of up to 20 unique tweets per bucket.

Outputs:
- bucket_tweet_samples_20_per_bucket.csv
- bucket_tweet_sample_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from collections import Counter, defaultdict, deque
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_INPUT_CSV = (
    PROJECT_ROOT
    / "final_results+visualisations_folder"
    / "combined_stance_results.csv"
)
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "bucket_tweet_samples_20_per_bucket.csv"
DEFAULT_SUMMARY_CSV = SCRIPT_DIR / "bucket_tweet_sample_summary.csv"

VALID_STANCES = {"favor", "against", "neutral"}
VALID_PARTIES = {"pro ruling", "pro opposition"}

KEYWORD_BUCKETS = {
    "Leader & Party Contestation": [
        "modi",
        "rahulgandhi",
        "congress",
    ],
    "Institutions, Democracy & State Accountability": [
        "democracy",
        "dictatorship",
        "spyware",
        "new parliament",
    ],
    "Economy, Development & Macro-Stewardship": [
        "aatmanirbhar",
        "demonetisation",
        "gdp",
        "inflation",
        "unemployment",
        "suicides",
    ],
    "Agrarian Reform & Farmer Movement": [
        "farm laws",
        "farmers protests",
        "msp",
    ],
    "Citizenship, Belonging & Mass Protest Politics": [
        "caa",
        "shaheen bagh",
    ],
    "Majoritarian Ideology & Hindu Nationalist Mobilization": [
        "hindutva",
        "sangh",
        "bhakts",
        "hindu",
    ],
    "Communal Relations, Minority Rights & Collective Violence": [
        "minorities",
        "muslim",
        "lynching",
        "sharia",
        "islamists",
        "hathras",
    ],
    "Symbolic Nationhood & Cultural-Religious Projects": [
        "ayodhya",
        "ram mandir",
        "mahotsav",
    ],
    "Security, Territory & Geopolitics": [
        "china",
        "kashmir",
        "balochistan",
        "kashmiri pandits",
    ],
}


def normalize_space(value: str | None) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text.strip())


def normalize_keyword(value: str | None) -> str:
    text = normalize_space(value).lower()
    text = text.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_tweet(value: str | None) -> str:
    return normalize_space(value).lower()


def get_bucket(keyword: str | None) -> str:
    keyword_lower = normalize_keyword(keyword)
    for bucket, keywords in KEYWORD_BUCKETS.items():
        for kw in keywords:
            if kw in keyword_lower:
                return bucket
    return "Other"


def canonical_bucket_keyword(bucket: str, keyword: str | None) -> str:
    keyword_lower = normalize_keyword(keyword)
    for kw in KEYWORD_BUCKETS.get(bucket, []):
        if kw in keyword_lower:
            return kw
    return keyword_lower or "unknown"


def load_candidates(input_csv: Path) -> tuple[dict, Counter]:
    candidates = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    seen_tweets_by_bucket = defaultdict(set)
    totals = Counter()

    with input_csv.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stance = normalize_space(row.get("fewshot_label")).lower()
            party = normalize_space(row.get("_label_norm")).lower()
            tweet = normalize_space(row.get("tweet"))
            keyword = normalize_keyword(row.get("keyword"))
            bucket = get_bucket(keyword)

            if stance not in VALID_STANCES:
                totals["skipped_invalid_stance"] += 1
                continue
            if party not in VALID_PARTIES:
                totals["skipped_invalid_party"] += 1
                continue
            if not tweet or not keyword:
                totals["skipped_missing_tweet_or_keyword"] += 1
                continue
            if bucket == "Other":
                totals["skipped_other_bucket"] += 1
                continue

            tweet_key = normalize_tweet(tweet)
            if tweet_key in seen_tweets_by_bucket[bucket]:
                totals["skipped_duplicate_tweet_within_bucket"] += 1
                continue
            seen_tweets_by_bucket[bucket].add(tweet_key)

            bucket_keyword = canonical_bucket_keyword(bucket, keyword)
            sample_row = {
                "bucket": bucket,
                "keyword": bucket_keyword,
                "source_keyword": keyword,
                "party": party,
                "stance": stance,
                "source_row": normalize_space(row.get("source_row")),
                "original_author": normalize_space(row.get("original_author")),
                "language": normalize_space(row.get("language")),
                "tweet_label": normalize_space(row.get("tweet_label")),
                "fewshot_reason": normalize_space(row.get("fewshot_reason")),
                "tweet": tweet,
            }
            candidates[bucket][bucket_keyword][(party, stance)].append(sample_row)
            totals["eligible_unique_tweets"] += 1

    return candidates, totals


def shuffle_candidates(candidates: dict, seed: int) -> None:
    rng = random.Random(seed)
    for keyword_map in candidates.values():
        for strata in keyword_map.values():
            for rows in strata.values():
                rng.shuffle(rows)


def pop_from_keyword(strata: dict, selected_strata_counts: Counter):
    available = [
        (selected_strata_counts[stratum], stratum, rows)
        for stratum, rows in strata.items()
        if rows
    ]
    if not available:
        return None

    _, stratum, rows = min(available, key=lambda item: (item[0], item[1][0], item[1][1]))
    selected_strata_counts[stratum] += 1
    return rows.pop()


def sample_bucket(bucket: str, keyword_map: dict, target: int) -> list[dict]:
    bucket_keywords = [kw for kw in KEYWORD_BUCKETS[bucket] if kw in keyword_map]
    extra_keywords = sorted(set(keyword_map) - set(bucket_keywords))
    keyword_order = bucket_keywords + extra_keywords

    selected: list[dict] = []
    selected_by_keyword = Counter()
    selected_strata_by_keyword = defaultdict(Counter)

    while len(selected) < target:
        added_this_round = 0
        keyword_order = sorted(keyword_order, key=lambda kw: (selected_by_keyword[kw], kw))

        for keyword in keyword_order:
            if len(selected) >= target:
                break
            row = pop_from_keyword(keyword_map[keyword], selected_strata_by_keyword[keyword])
            if row is None:
                continue
            row["bucket_sample_number"] = len(selected) + 1
            selected.append(row)
            selected_by_keyword[keyword] += 1
            added_this_round += 1

        if added_this_round == 0:
            break

    return selected


def write_samples(samples: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "bucket",
        "keyword",
        "stance",
        "tweet_label",
        "tweet",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(samples)


def write_summary(samples: list[dict], summary_csv: Path) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter((row["bucket"], row["keyword"], row["party"], row["stance"]) for row in samples)
    columns = ["bucket", "keyword", "party", "stance", "sample_count"]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for (bucket, keyword, party, stance), count in sorted(counts.items()):
            writer.writerow(
                {
                    "bucket": bucket,
                    "keyword": keyword,
                    "party": party,
                    "stance": stance,
                    "sample_count": count,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--tweets-per-bucket", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates, totals = load_candidates(args.input_csv)
    shuffle_candidates(candidates, args.seed)

    all_samples: list[dict] = []
    for bucket in KEYWORD_BUCKETS:
        bucket_samples = sample_bucket(bucket, candidates.get(bucket, {}), args.tweets_per_bucket)
        all_samples.extend(bucket_samples)

    write_samples(all_samples, args.output_csv)
    write_summary(all_samples, args.summary_csv)

    print(f"Input: {args.input_csv}")
    print(f"Eligible unique tweets: {totals['eligible_unique_tweets']:,}")
    print(f"Sampled tweets: {len(all_samples):,}")
    print(f"Wrote: {args.output_csv}")
    print(f"Wrote: {args.summary_csv}")

    by_bucket = Counter(row["bucket"] for row in all_samples)
    print("\nSamples per bucket:")
    for bucket in KEYWORD_BUCKETS:
        print(f"  {bucket}: {by_bucket[bucket]}")


if __name__ == "__main__":
    main()
