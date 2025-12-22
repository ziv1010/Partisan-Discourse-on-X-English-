#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract tweet subjects/keywords with KeyBERT.

- Input CSV columns: timestamp,tweet,retweet_author,original_author,tweet_label
- Outputs the same CSV + subjects (list[str]) + subjects_scored (list of {text, score})

Usage:
  python extract_subjects_keybert.py --in tweets.csv --out tweets_with_subjects.csv \
      --model paraphrase-multilingual-MiniLM-L12-v2 --top_n 3
"""

import argparse, json, re, sys
from typing import List, Tuple
import pandas as pd
from tqdm import tqdm

# --- preprocessing helpers (URLs, mentions, hashtags) ---
URL_RE      = re.compile(r"https?://\S+")
MENTION_RE  = re.compile(r"@([A-Za-z0-9_]{1,50})")
HASHTAG_RE  = re.compile(r"#([A-Za-z0-9_]+)")
WS_RE       = re.compile(r"\s+")

def try_hashtag_segmenter():
    """Try to load 'wordsegment' for hashtag segmentation; fallback to identity."""
    try:
        from wordsegment import load, segment
        load()

        def seg_fun(tag: str) -> str:
            # wordsegment returns a list of tokens; join with space
            return " ".join(segment(tag))
        return seg_fun
    except Exception:
        return lambda tag: tag  # no-op fallback

SEG_HASHTAG = try_hashtag_segmenter()

def normalize_hashtags(text: str) -> str:
    # Replace hashtags with segmented text (e.g., '#aatmanirbharbharat' -> 'aatmanirbhar bharat')
    def _repl(m):
        tag = m.group(1)
        return SEG_HASHTAG(tag.lower())
    return HASHTAG_RE.sub(lambda m: _repl(m), text)

def strip_urls_mentions(text: str) -> str:
    # Remove URLs; replace @handles with the handle token (without '@') to keep subject signal
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(lambda m: m.group(1), text)  # keep 'narendramodi' as token
    text = WS_RE.sub(" ", text).strip()
    return text

def preprocess_tweet(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    text = raw.replace("\u2066", "").replace("\u2069", "")  # strip directional isolates if present
    text = normalize_hashtags(text)
    text = strip_urls_mentions(text)
    return text

# --- KeyBERT setup ---
def build_keybert(model_name: str):
    # Using SBERT backends (multilingual by default for your data)
    from keybert import KeyBERT
    if model_name:
        return KeyBERT(model=model_name)
    return KeyBERT()  # defaults to a sensible SBERT model internally

def extract_keywords_doc(kw_model, text: str, top_n=3) -> List[Tuple[str, float]]:
    """
    KeyBERT extraction with 1–2-gram candidates (unigrams + bigrams) + MMR diversity.
    """
    if not text:
        return []
    # No stopword list to keep non-English subject terms; adjust if you want English-only.
    res = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),  # <-- only unigrams and bigrams
        stop_words=None,
        use_mmr=True,           # diversify to avoid near-duplicates
        diversity=0.7,
        top_n=top_n
    )
    return [(kw.strip(), float(score)) for kw, score in res]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp", required=True, help="Input CSV path")
    ap.add_argument("--out", dest="out", required=True, help="Output CSV path")
    ap.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2",
                    help="Sentence-Transformers model for KeyBERT backend")
    ap.add_argument("--top_n", type=int, default=3, help="Number of keywords per tweet")
    ap.add_argument("--batch", type=int, default=0, help="(optional) rows to process; 0=all")
    args = ap.parse_args()

    try:
        df = pd.read_csv(args.inp)
    except Exception as e:
        print(f"Failed to read CSV: {e}", file=sys.stderr); sys.exit(1)

    # Ensure required cols exist
    for col in ["tweet"]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV.")

    # Build model
    kw_model = build_keybert(args.model)

    # Optionally subset
    if args.batch and args.batch > 0:
        df = df.iloc[: args.batch].copy()

    # Preprocess + extract
    subjects_list = []
    subjects_scored = []

    for txt in tqdm(df["tweet"].astype(str).tolist(), desc="Extracting subjects"):
        clean = preprocess_tweet(txt)
        kws = extract_keywords_doc(kw_model, clean, top_n=args.top_n)
        subjects_scored.append([{"text": k, "score": s} for k, s in kws])
        subjects_list.append([k for k, _ in kws])

    df["subjects"] = subjects_list
    df["subjects_scored"] = subjects_scored

    df.to_csv(args.out, index=False)
    print(f"Wrote: {args.out}")

if __name__ == "__main__":
    main()
