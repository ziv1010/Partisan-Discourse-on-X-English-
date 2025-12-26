"""
Keyword Tweet Extractor (Custom List) - V2

This script extracts ALL tweets for a specific keyword list.

Features:
- **Primary search**: exact match on `keyword` column (case-insensitive)
- **Text search**: case-insensitive text search in tweet content
- **ALWAYS uses BOTH searches** to extract maximum tweets
- **NEW Dedup logic**: Same tweet CAN appear for different keywords
- **Per-keyword dedup**: Same tweet + same author NOT allowed within same keyword
"""

import pandas as pd
import re
import os
from pathlib import Path
from functools import reduce
import operator

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data paths
CSV_PATH = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/final_data/tweets_exploded_by_keyword.csv"
OUT_DIR = Path("extracted_by_keyword")

SEED = 42

# ============================================================================
# KEYWORD LIST - Your specified keywords
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
    "ratetvdebate"
]

# Column configuration
POSSIBLE_TWEET_COLS = ("tweet", "text", "full_text", "content", "body")
KEYWORD_COL = "keyword"
LABEL_COL = "tweet_label"
TARGETS = ["pro ruling", "pro opposition"]

print(f"CSV Path: {CSV_PATH}")
print(f"Output Dir: {OUT_DIR}")
print(f"Total keywords: {len(KEYWORDS)}")
print(f"Mode: EXTRACT ALL (same tweet can appear for different keywords)")
print(f"Dedup Rule: Same tweet + same author NOT allowed within same keyword")


# ---------- Helper functions ----------
def _norm_nospace(x):
    """Lowercase + drop all non-alphanumerics (incl. spaces). Case-insensitive."""
    if isinstance(x, pd.Series):
        return (
            x.fillna("")
             .astype(str)
             .str.lower()  # Case-insensitive: 'RAM' and 'ram' become 'ram'
             .str.replace(r"[^a-z0-9]+", "", regex=True)
        )
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def _phrase_variants(s: str) -> list:
    """
    Support ' or ' and '|' as OR separators inside a keyword/phrase.
    Returns the ORIGINAL (lowercased/trimmed) variants.
    """
    raw = str(s).strip()
    parts = re.split(r"\s+or\s+|\|", raw, flags=re.IGNORECASE)
    parts = [p.strip().lower() for p in parts if p.strip()]
    return parts if parts else [raw.lower().strip()]


def _any_contains_norm(tw_norm_series: pd.Series, raw_phrase: str) -> pd.Series:
    """
    Build a boolean mask: tweet contains ANY normalized variant of raw_phrase.
    Case-insensitive text search.
    
    Since tw_norm_series is already normalized (lowercased, non-alphanum removed),
    both 'RAM' and 'ram' will match 'ram' in the search.
    """
    variants = _phrase_variants(raw_phrase)
    variants_norm = [_norm_nospace(v) for v in variants]
    masks = [tw_norm_series.str.contains(re.escape(vn), regex=True) for vn in variants_norm]
    return reduce(operator.or_, masks) if masks else pd.Series(False, index=tw_norm_series.index)


# ---------- Load & prep ----------
print("Loading CSV... (this may take a while for large files)")
df = pd.read_csv(CSV_PATH, low_memory=False)
print(f"Loaded {len(df):,} rows")
print(f"Columns: {df.columns.tolist()}")

# choose tweet column
tweet_col = next((c for c in POSSIBLE_TWEET_COLS if c in df.columns), None)
if tweet_col is None:
    raise ValueError(f"Couldn't find a tweet/text column. Tried: {POSSIBLE_TWEET_COLS}.")
print(f"Tweet column: {tweet_col}")

# stable id
id_col = "source_row" if "source_row" in df.columns else None
if id_col is None:
    df["source_row"] = df.index
    id_col = "source_row"

# Determine author column (prefer original_author, fallback to retweet_author)
author_col = None
for col in ['original_author', 'retweet_author', 'author', 'user']:
    if col in df.columns:
        author_col = col
        break
if author_col is None:
    print("WARNING: No author column found, using source_row as fallback")
    author_col = id_col
print(f"Author column: {author_col}")

# Create composite key for deduplication: tweet_text + author
# This allows same tweet from different authors
df['_tweet_author_key'] = df[tweet_col].astype(str) + '|||' + df[author_col].astype(str).fillna('')

# NOTE: We do NOT dedupe globally here anymore
# Deduplication is now per-keyword based on (tweet, author) pairs
print(f"Total rows available: {len(df):,}")


# normalize labels to TARGETS
def normalize_label(x: str) -> str:
    if not isinstance(x, str): return "other"
    s = x.strip().lower()
    if re.search(r"\bpro[-_\s]*rul(?:ing)?\b", s): return "pro ruling"
    if re.search(r"\bpro[-_\s]*(opp|opposition)\b", s): return "pro opposition"
    return "other"


df["_label_norm"] = df[LABEL_COL].apply(normalize_label)
print(f"Label distribution (before filtering):")
print(df["_label_norm"].value_counts())

df = df[df["_label_norm"].isin(TARGETS)].copy()
print(f"\nAfter filtering to TARGETS: {len(df):,} rows")

# lowercase keyword col for primary match
if KEYWORD_COL not in df.columns:
    raise ValueError(f"Column '{KEYWORD_COL}' not found. Available: {list(df.columns)[:25]}")

df["_kw_lc"] = df[KEYWORD_COL].astype(str).str.strip().str.lower()

# normalized tweet text for text search (case-insensitive)
print("Normalizing tweet text for text search (case-insensitive)...")
tw_norm = _norm_nospace(df[tweet_col])
print("Done.")


# Per-keyword tracking: tracks (tweet_text, author) pairs used for each keyword
# This allows the SAME tweet to be used for DIFFERENT keywords
# But prevents same tweet+author combo within the same keyword
PER_KEYWORD_USED = {}  # keyword -> set of (tweet, author) keys


def extract_all_for_keyword(kw_raw: str) -> tuple:
    """
    Extract ALL tweets for a keyword using BOTH search methods:
    1. PRIMARY: Exact match on keyword column (case-insensitive)
    2. TEXT SEARCH: Case-insensitive text search in tweet content
    
    ALWAYS combines both methods to get maximum tweets.
    
    NEW Deduplication logic:
    - Same tweet CAN appear for DIFFERENT keywords
    - Same tweet + same author is NOT allowed within the SAME keyword
    - Same tweet + different author IS allowed within the same keyword
    
    Returns:
        (DataFrame of extracted tweets, stats dict)
    """
    global PER_KEYWORD_USED
    
    # variants for this bucket (handles 'or' and '|' separators)
    kw_variants = _phrase_variants(kw_raw)
    canonical = kw_variants[0] if kw_variants else str(kw_raw).strip().lower()
    
    # Initialize tracking set for this keyword
    if canonical not in PER_KEYWORD_USED:
        PER_KEYWORD_USED[canonical] = set()

    # PRIMARY pool = keyword column equals any variant (case-insensitive)
    pool_primary = df[df["_kw_lc"].isin(kw_variants)].copy()
    primary_count_raw = len(pool_primary)
    
    # TEXT SEARCH pool = tweet text contains ANY normalized variant (case-insensitive)
    contains_any = _any_contains_norm(tw_norm, kw_raw)
    pool_text_search = df[contains_any].copy()
    
    # Combine: primary + text search
    primary_keys = set(pool_primary['_tweet_author_key'])
    pool_text_new = pool_text_search[~pool_text_search['_tweet_author_key'].isin(primary_keys)]
    
    out_kw = pd.concat([pool_primary, pool_text_new], axis=0)
    text_search_count_raw = len(pool_text_new)
    
    # Remove duplicates by (tweet, author) within this keyword extraction
    # This allows same tweet with different authors
    out_kw = out_kw.drop_duplicates(subset=['_tweet_author_key']).copy()
    
    # Track what we used for this keyword
    PER_KEYWORD_USED[canonical] |= set(out_kw['_tweet_author_key'])
    
    # Shuffle for variety
    if not out_kw.empty:
        out_kw = out_kw.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    # Overwrite keyword column with canonical keyword
    out_kw[KEYWORD_COL] = canonical

    # Compute stats per label
    primary_count = len(pool_primary.drop_duplicates(subset=['_tweet_author_key']))
    text_search_count = len(out_kw) - primary_count
    
    stats = {
        "total_extracted": len(out_kw),
        "from_primary": primary_count,
        "from_text_search": max(0, text_search_count),
        "by_label": {}
    }
    for label in TARGETS:
        label_count = len(out_kw[out_kw["_label_norm"] == label])
        stats["by_label"][label] = label_count

    return out_kw, stats


# --- Run extraction for all keywords ---
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Reset per-keyword tracking
PER_KEYWORD_USED = {}

combined = []
reports = {}

print("=" * 80)
print("EXTRACTING ALL TWEETS BY KEYWORD (PRIMARY + TEXT SEARCH)")
print("NEW: Same tweet can appear for different keywords")
print("NEW: Same tweet + same author prevented within each keyword")
print(f"Keywords: {KEYWORDS}")
print("=" * 80)

for kw in KEYWORDS:
    out_kw, stat_kw = extract_all_for_keyword(kw)

    combined.append(out_kw)
    reports[kw] = stat_kw

    # write per-keyword files
    cols_out = [id_col, tweet_col, author_col, LABEL_COL, "_label_norm", KEYWORD_COL, "subjects_scored"]
    cols_out = [c for c in cols_out if c in out_kw.columns]
    canonical_name = _phrase_variants(kw)[0].replace(" ", "_")
    out_csv = OUT_DIR / f"extracted_{canonical_name}.csv"
    out_ids = OUT_DIR / f"extracted_{canonical_name}_ids.txt"

    out_kw[cols_out].to_csv(out_csv, index=False)
    with open(out_ids, "w", encoding="utf-8") as f:
        for v in out_kw[id_col].tolist():
            f.write(f"{v}\n")

    # Status with breakdown
    print(f"[OK] '{kw}'")
    print(f"     -> total: {stat_kw['total_extracted']}, primary: {stat_kw['from_primary']}, text_search: {stat_kw['from_text_search']}")
    print(f"     -> by label: {stat_kw['by_label']}")

print("\n" + "=" * 80)


# Combined outputs
all_out = pd.concat(combined, axis=0).reset_index(drop=True) if combined else pd.DataFrame()
cols_out_all = [id_col, tweet_col, author_col, LABEL_COL, "_label_norm", KEYWORD_COL, "subjects_scored"]
cols_out_all = [c for c in cols_out_all if c in all_out.columns]

total_rows = len(all_out)
all_csv = OUT_DIR / f"extracted_ALL_{len(KEYWORDS)}keywords_{total_rows}rows.csv"
all_ids = OUT_DIR / f"extracted_ALL_{len(KEYWORDS)}keywords_ids.txt"

all_out[cols_out_all].to_csv(all_csv, index=False)
with open(all_ids, "w", encoding="utf-8") as f:
    for v in all_out[id_col].tolist():
        f.write(f"{v}\n")

print(f"[OK] Combined: {all_csv} (rows={total_rows})")


# Summary report
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\nMode: EXTRACT ALL (same tweet can appear for different keywords)")
print(f"Dedup: Same tweet + same author prevented within each keyword\n")

# Create summary table
summary_data = []
for kw, stat in reports.items():
    summary_data.append({
        "keyword": kw,
        "total": stat['total_extracted'],
        "primary": stat['from_primary'],
        "text_search": stat['from_text_search'],
        "pro_ruling": stat['by_label'].get('pro ruling', 0),
        "pro_opposition": stat['by_label'].get('pro opposition', 0)
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print(f"\n✅ All files saved to: {OUT_DIR}/")
print(f"   Total rows extracted: {total_rows:,}")
print(f"   Keywords processed: {len(KEYWORDS)}")

# Calculate overlap stats (how many tweets appear in multiple keywords)
tweet_keyword_counts = all_out.groupby(tweet_col)[KEYWORD_COL].nunique()
multi_keyword_tweets = (tweet_keyword_counts > 1).sum()
print(f"   Tweets appearing in multiple keywords: {multi_keyword_tweets:,}")

# Save summary to CSV
summary_csv = OUT_DIR / "extraction_summary.csv"
summary_df.to_csv(summary_csv, index=False)
print(f"   Summary saved to: {summary_csv}")
