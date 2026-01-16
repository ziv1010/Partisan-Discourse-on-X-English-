#!/usr/bin/env python3
# make_eval_500.py
# Freeze a 500-row evaluation slice for sentiment experiments.
# - Includes all available labeled rows (via --label-col) up to N.
# - Fills the remainder with stratified samples over (script, has_url, has_hashtag).
# - Drops exact duplicate texts and RT lines ("RT @...").
# - Saves both eval_500.csv and eval_500_ids.txt (original row indices, tweet_id if present).
#
# Usage:
#   python make_eval_500.py --input master_tweets.csv --text-col tweet --label-col sentiment #       --n 500 --out-csv eval_500.csv --out-ids eval_500_ids.txt --seed 42
#
# Notes:
# - 'script' is computed by Devanagari character ratio; threshold default = 0.4.
# - If labeled rows exceed N, a stratified sample from labeled is taken to size N.
# - If 'tweet_id' exists, IDs file includes it; otherwise only original indices.
#
import argparse, math, re, sys
from typing import List, Tuple
import numpy as np
import pandas as pd

def devanagari_ratio(text: str) -> float:
    s = str(text) if text is not None else ""
    if not s:
        return 0.0
    count = sum(1 for ch in s if '\u0900' <= ch <= '\u097F')
    return count / len(s)

def add_helper_columns(df: pd.DataFrame, text_col: str, script_thresh: float) -> pd.DataFrame:
    out = df.copy()
    s = out[text_col].astype(str)
    out["has_url"] = s.str.contains(r"http[s]?://", regex=True, na=False)
    out["has_hashtag"] = s.str.contains(r"#\w+", regex=True, na=False)
    out["has_emoji"] = s.str.contains(r"[\U0001F300-\U0001FAFF]", regex=True, na=False)
    out["len_words"] = s.str.split().str.len()
    out["devanagari_ratio"] = s.apply(devanagari_ratio)
    out["script"] = out["devanagari_ratio"].map(lambda x: "devanagari" if x >= script_thresh else "roman_hindi")
    return out

def drop_dupes_and_rts(df: pd.DataFrame, text_col: str, drop_rts: bool=True) -> pd.DataFrame:
    out = df.copy()
    if drop_rts:
        out = out[~out[text_col].astype(str).str.startswith("RT @", na=False)]
    # Drop exact duplicate text rows
    out = out.drop_duplicates(subset=[text_col], keep="first")
    # If tweet_id exists, drop duplicate IDs too
    if "tweet_id" in out.columns:
        out = out.drop_duplicates(subset=["tweet_id"], keep="first")
    return out

def stratified_sample(df: pd.DataFrame, strata_cols: List[str], n: int, seed: int) -> pd.DataFrame:
    """Proportional allocation over given strata; rounds down then distributes remainder by largest fractional part."""
    if n <= 0 or df.empty:
        return df.head(0).copy()
    groups = df.groupby(strata_cols, dropna=False)
    counts = groups.size()
    total = int(counts.sum())
    if total == 0:
        return df.head(0).copy()
    ideal = counts / total * n
    alloc = np.floor(ideal).astype(int)
    remainder = n - int(alloc.sum())
    if remainder > 0:
        frac = ideal - alloc
        order = list(frac.sort_values(ascending=False).index)
        for key in order[:remainder]:
            alloc.loc[key] += 1
    parts = []
    rng = np.random.RandomState(seed)
    for key, group in groups:
        k = int(alloc.get(key, 0))
        if k <= 0:
            continue
        if len(group) <= k:
            parts.append(group)
        else:
            parts.append(group.sample(n=k, random_state=rng))
    if not parts:
        return df.sample(n=min(n, len(df)), random_state=rng)
    out = pd.concat(parts, axis=0)
    # In rare cases overshoot due to tiny groups; trim
    if len(out) > n:
        out = out.sample(n=n, random_state=rng)
    # Shuffle once
    out = out.sample(frac=1.0, random_state=rng).reset_index(drop=True)
    return out

def main():
    ap = argparse.ArgumentParser(description="Freeze a 500-row eval slice with stratification and ID manifest.")
    ap.add_argument("--input", required=True, help="Path to master CSV of tweets")
    ap.add_argument("--text-col", default="tweet", help="Text column name (default: tweet)")
    ap.add_argument("--label-col", default="sentiment", help="Gold label column (default: sentiment). Set to '' if none.")
    ap.add_argument("--n", type=int, default=500, help="Target eval size (default: 500)")
    ap.add_argument("--out-csv", default="eval_500.csv", help="Output CSV path for frozen eval slice")
    ap.add_argument("--out-ids", default="eval_500_ids.txt", help="Output text/CSV file of original row IDs (and tweet_id if available)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    ap.add_argument("--script-thresh", type=float, default=0.4, help="Devanagari ratio threshold (default: 0.4)")
    ap.add_argument("--keep-rt", action="store_true", help="Keep retweets (by default RTs are dropped)")
    args = ap.parse_args()

    # Load
    df = pd.read_csv(args.input)
    if args.text_col not in df.columns:
        sys.exit(f"[ERROR] text column '{args.text_col}' not found. Available: {list(df.columns)}")

    # Preserve original index for ID manifest
    df = df.copy()
    df["orig_index"] = df.index

    # Drop RTs/dupes
    df = drop_dupes_and_rts(df, text_col=args.text_col, drop_rts=not args.keep_rt)

    # Add helper columns
    df = add_helper_columns(df, text_col=args.text_col, script_thresh=args.script_thresh)

    # Identify labeled rows (if label column exists and is not empty string)
    label_col = args.label_col if args.label_col and args.label_col in df.columns else None
    if label_col is None:
        labeled = df.head(0).copy()
    else:
        labeled = df[~df[label_col].isna()].copy()

    # If labeled > N, take a stratified sample from labeled only
    N = int(args.n)
    if len(labeled) > 0 and len(labeled) > N:
        labeled = stratified_sample(
            labeled,
            strata_cols=["script", "has_url", "has_hashtag"],
            n=N,
            seed=args.seed
        )
        remainder_needed = 0
    else:
        remainder_needed = max(0, N - len(labeled))

    # Unlabeled pool excludes already selected labeled rows
    if len(labeled) > 0:
        selected_ids = set(labeled["orig_index"].tolist())
        unlabeled = df[~df["orig_index"].isin(selected_ids)].copy()
    else:
        unlabeled = df.copy()

    # Fill remainder with stratified sample from unlabeled
    if remainder_needed > 0:
        fill = stratified_sample(
            unlabeled,
            strata_cols=["script", "has_url", "has_hashtag"],
            n=remainder_needed,
            seed=args.seed
        )
        eval_df = pd.concat([labeled, fill], axis=0, ignore_index=True)
    else:
        eval_df = labeled.copy()

    # Final shuffle & trim just in case
    rng = np.random.RandomState(args.seed)
    if len(eval_df) > N:
        eval_df = eval_df.sample(n=N, random_state=rng)
    else:
        eval_df = eval_df.sample(frac=1.0, random_state=rng)

    # Save outputs
    eval_df.to_csv(args.out_csv, index=False)

    # IDs manifest
    id_cols = ["orig_index"]
    if "tweet_id" in eval_df.columns:
        id_cols.append("tweet_id")
    eval_df[id_cols].to_csv(args.out_ids, index=False)

    # Brief report
    # Count by script/url/hashtag to verify stratification roughly preserved
    summary = eval_df.groupby(["script", "has_url", "has_hashtag"]).size().reset_index(name="count")
    print(f"[OK] Wrote eval slice: {args.out_csv}  (rows={len(eval_df)})")
    print(f"[OK] Wrote ID manifest: {args.out_ids}")
    print("\n[Strata counts]\n", summary.sort_values(["script","has_url","has_hashtag","count"], ascending=[True,False,False,False]).to_string(index=False))

if __name__ == "__main__":
    main()
