# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# Split every CSV in a folder into 80:20 (per-file) and build master_train.csv / master_test.csv.

# - Attempts stratified split by 'stance' when feasible (>=2 classes and enough rows);
#   falls back to non-stratified random split otherwise.
# - Preserves all columns and their order.
# - Deterministic with --seed (default 42).
# - Skips empty files gracefully and reports a summary.

# Usage:
#   python split_to_master.py /path/to/folder \
#       --train-out /path/to/master_train.csv \
#       --test-out /path/to/master_test.csv \
#       --seed 42
# """

# import argparse
# import math
# from pathlib import Path
# import sys
# import pandas as pd
# import numpy as np

# def read_csv_robust(p: Path) -> pd.DataFrame:
#     """
#     Read CSV robustly (handles stray commas/quotes and UTF-8 BOM).
#     Returns empty DataFrame if file is unreadable.
#     """
#     for enc in ("utf-8-sig", "utf-8", "latin1"):
#         try:
#             # engine='python' is more forgiving with quotes/commas inside text
#             df = pd.read_csv(p, engine="python", dtype=str)
#             # Drop fully-empty rows (all NaN)
#             df = df.dropna(how="all")
#             # Normalize column whitespace
#             df.columns = [c.strip() for c in df.columns]
#             return df
#         except Exception:
#             continue
#     print(f"[WARN] Could not read {p.name} with common encodings. Skipping.", file=sys.stderr)
#     return pd.DataFrame()

# def stratified_split(df: pd.DataFrame, test_frac: float, seed: int, label_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     Try per-class 80:20. If any class is too small, fallback to global split.
#     """
#     if label_col not in df.columns:
#         raise ValueError("Label column not present")

#     series = df[label_col].astype(str).fillna("")

#     # Require at least 2 distinct classes and total rows >= 5 for sensible splits
#     classes = series.unique()
#     if len(classes) < 2 or len(df) < 5:
#         raise ValueError("Not suitable for stratified split")

#     # Each class should have at least 2 rows to allow a test pick eventually
#     grp_sizes = series.value_counts()
#     if (grp_sizes < 2).any():
#         raise ValueError("Some classes too small for stratified split")

#     rng = np.random.RandomState(seed)
#     test_idx = []

#     for cls, gdf in df.groupby(series, sort=False):
#         n = len(gdf)
#         n_test = max(1, int(math.floor(test_frac * n)))
#         # If a group is extremely tiny (e.g., n=2), we’ll still take 1 to keep representation
#         pick = rng.choice(gdf.index.values, size=min(n_test, n-1) if n > 1 else 1, replace=False)
#         test_idx.extend(pick.tolist())

#     test_mask = df.index.isin(test_idx)
#     df_test = df.loc[test_mask]
#     df_train = df.loc[~test_mask]

#     # Safety: ensure neither is empty; if so, fallback to global
#     if df_train.empty or df_test.empty:
#         raise ValueError("Stratified split produced empty partition")

#     return df_train, df_test

# def random_split(df: pd.DataFrame, test_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
#     rng = np.random.RandomState(seed)
#     idx = df.index.values
#     rng.shuffle(idx)
#     n_test = max(1, int(math.floor(test_frac * len(idx)))) if len(idx) > 1 else 1
#     test_idx = set(idx[:n_test])
#     df_test = df.loc[df.index.isin(test_idx)]
#     df_train = df.loc[~df.index.isin(test_idx)]
#     # Safety: if one side is empty (tiny files), move one row
#     if df_train.empty and not df_test.empty:
#         df_train = df_test.iloc[:1]
#         df_test = df_test.iloc[1:]
#     if df_test.empty and not df_train.empty and len(df_train) > 1:
#         df_test = df_train.iloc[:1]
#         df_train = df_train.iloc[1:]
#     return df_train, df_test

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("input_dir", type=str, help="Folder containing many CSVs")
#     ap.add_argument("--train-out", type=str, default="master_train.csv", help="Output master train CSV")
#     ap.add_argument("--test-out", type=str, default="master_test.csv", help="Output master test CSV")
#     ap.add_argument("--test-frac", type=float, default=0.2, help="Test fraction per file (default 0.2)")
#     ap.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
#     ap.add_argument("--label-col", type=str, default="stance", help="Label column for optional stratification (default 'stance')")
#     args = ap.parse_args()

#     input_dir = Path(args.input_dir).expanduser().resolve()
#     if not input_dir.exists() or not input_dir.is_dir():
#         print(f"[ERR] Input directory not found: {input_dir}", file=sys.stderr)
#         sys.exit(1)

#     csv_paths = sorted([p for p in input_dir.glob("*.csv") if p.is_file()])
#     if not csv_paths:
#         print(f"[ERR] No CSV files found in {input_dir}", file=sys.stderr)
#         sys.exit(1)

#     all_train, all_test = [], []
#     total_in, total_train, total_test = 0, 0, 0

#     print(f"[INFO] Found {len(csv_paths)} CSVs. Processing with per-file 80:20 splits...")
#     for p in csv_paths:
#         df = read_csv_robust(p)
#         if df.empty:
#             print(f"[WARN] {p.name}: empty or unreadable, skipping.")
#             continue

#         # Normalize column whitespace in cells (optional light cleanup)
#         df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

#         # Attempt stratified split on label-col; fallback to random split if not feasible
#         try:
#             df_train, df_test = stratified_split(df, args.test_frac, args.seed, args.label_col)
#             mode = "stratified"
#         except Exception:
#             df_train, df_test = random_split(df, args.test_frac, args.seed)
#             mode = "random"

#         all_train.append(df_train)
#         all_test.append(df_test)

#         n_in = len(df)
#         n_tr = len(df_train)
#         n_te = len(df_test)
#         total_in += n_in
#         total_train += n_tr
#         total_test += n_te

#         print(f"[OK] {p.name}: {mode:11s}  total={n_in:5d}  train={n_tr:5d}  test={n_te:5d}")

#     if not all_train and not all_test:
#         print("[ERR] Nothing to write. Exiting.", file=sys.stderr)
#         sys.exit(1)

#     # Concatenate and write
#     train_out = Path(args.train_out).expanduser().resolve()
#     test_out = Path(args.test_out).expanduser().resolve()

#     # Align columns across files (outer union), keep order from first seen file
#     def concat_align(dfs):
#         if not dfs:
#             return pd.DataFrame()
#         # Start with columns of the first df, then union
#         first_cols = list(dfs[0].columns)
#         union_cols = list(first_cols)
#         for d in dfs[1:]:
#             for c in d.columns:
#                 if c not in union_cols:
#                     union_cols.append(c)
#         aligned = [d.reindex(columns=union_cols) for d in dfs]
#         return pd.concat(aligned, axis=0, ignore_index=True)

#     master_train = concat_align(all_train)
#     master_test = concat_align(all_test)

#     master_train.to_csv(train_out, index=False, encoding="utf-8")
#     master_test.to_csv(test_out, index=False, encoding="utf-8")

#     print("\n[SUMMARY]")
#     print(f"  Files processed : {len(csv_paths)}")
#     print(f"  Total rows in   : {total_in}")
#     print(f"  Master train    : {len(master_train)} → {train_out}")
#     print(f"  Master test     : {len(master_test)} → {test_out}")
#     print("  Done.")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Per-keyword JSONs (10 items each with ~3 pos / 3 neg / 3 neutral + 1 filler),
then 85:15 split on the REMAINDER.

- Reads all CSVs in a folder (robust reader), outer-union aligns columns.
- Ignores/drops any column canonically named "keyword_cat" or "label_norm" (some inputs have them, some don't).
- Writes one JSON per keyword to --json-outdir named: <prefix>_<keyword>_stance.json
    with objects: {"entity","statement","stance","reason"}.
    entity=keyword, statement=tweet, stance in {positive,negative,neutral}, reason from stance reason/_.
- Deterministic with --seed. Each keyword selection is independently seeded (stable hash).
- After removing ALL selected rows for ALL keywords, splits remainder 85:15.
"""

import argparse
import json
import math
import re
import hashlib
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# -------------------- Robust CSV reading & alignment --------------------

def read_csv_robust(p: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            df = pd.read_csv(p, engine="python", dtype=str)
            df = df.dropna(how="all")
            df.columns = [str(c).strip() for c in df.columns]
            # Drop any variant of columns named "keyword_cat" or "label_norm" (case/spacing insensitive)
            # so they aren't accidentally used or unioned across files.
            drop_cols = [c for c in df.columns if _canon(c) in {"keywordcat", "labelnorm"}]
            if drop_cols:
                df = df.drop(columns=drop_cols, errors="ignore")
            return df
        except Exception:
            continue
    print(f"[WARN] Could not read {p.name} with common encodings. Skipping.", file=sys.stderr)
    return pd.DataFrame()

def concat_align(dfs):
    dfs = [d for d in dfs if not d.empty]
    if not dfs: return pd.DataFrame()
    first_cols = list(dfs[0].columns)
    union_cols = list(first_cols)
    for d in dfs[1:]:
        for c in d.columns:
            if c not in union_cols:
                union_cols.append(c)
    return pd.concat([d.reindex(columns=union_cols) for d in dfs], axis=0, ignore_index=True)

# -------------------- Column finding (tolerant) --------------------

def _canon(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', s.lower()) if s is not None else ''

def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    # map canonical header -> original
    canon_map = {_canon(c): c for c in df.columns}
    for cand in candidates:
        ckey = _canon(cand)
        if ckey in canon_map:
            return canon_map[ckey]
    # soft contains (e.g., 'stance reason ,')
    for c in df.columns:
        if any(_canon(cand) in _canon(c) for cand in candidates):
            return c
    return None

# -------------------- Stance mapping --------------------

POS_LABELS = {
    "for","pro","proruling","support","supports","favour","favor","positive",
    "pro_ruling","pro-ruling","pro ruling"
}
NEG_LABELS = {
    "against","anti","con","antiruling","oppose","opposes","negative",
    "anti_ruling","anti-ruling","anti ruling"
}
NEU_LABELS = {"neutral","mixed","unknown","undecided"}

def clean_str(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return ""
    return str(x).strip()

def normalize_stance(raw: str) -> str:
    s = _canon(clean_str(raw))
    if s in POS_LABELS: return "positive"
    if s in NEG_LABELS: return "negative"
    if s in NEU_LABELS: return "neutral"
    if s == "for": return "positive"
    if s == "against": return "negative"
    if s == "neutral": return "neutral"
    return s or "unknown"

# -------------------- Splitting helpers --------------------

def stratified_split(df: pd.DataFrame, test_frac: float, seed: int, label_col: str):
    if label_col not in df.columns:
        raise ValueError("Label column not present")
    series = df[label_col].astype(str).fillna("")
    if len(series.unique()) < 2 or len(df) < 5:
        raise ValueError("Not suitable for stratified split")
    if (series.value_counts() < 2).any():
        raise ValueError("Some classes too small for stratified split")

    rng = np.random.RandomState(seed)
    test_idx = []
    for cls, gdf in df.groupby(series, sort=False):
        n = len(gdf)
        if n == 1: continue
        n_test = max(1, int(math.floor(test_frac * n)))
        pick = rng.choice(gdf.index.values, size=min(n_test, n-1), replace=False)
        test_idx.extend(pick.tolist())

    test_mask = df.index.isin(test_idx)
    df_test = df.loc[test_mask]
    df_train = df.loc[~test_mask]
    if df_train.empty or df_test.empty:
        raise ValueError("Stratified split produced empty partition")
    return df_train, df_test

def random_split(df: pd.DataFrame, test_frac: float, seed: int):
    rng = np.random.RandomState(seed)
    idx = df.index.values.copy()
    rng.shuffle(idx)
    n_test = max(1, int(math.floor(test_frac * len(idx)))) if len(idx) > 1 else 1
    test_idx = set(idx[:n_test])
    df_test = df.loc[df.index.isin(test_idx)]
    df_train = df.loc[~df.index.isin(test_idx)]
    if df_train.empty and not df_test.empty:
        df_train = df_test.iloc[:1]; df_test = df_test.iloc[1:]
    if df_test.empty and not df_train.empty and len(df_train) > 1:
        df_test = df_train.iloc[:1]; df_train = df_train.iloc[1:]
    return df_train, df_test

# -------------------- Per-keyword JSON selection --------------------

def stable_rng(seed: int, key: str) -> np.random.RandomState:
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)
    return np.random.RandomState(seed ^ h)

def pick_json_rows_per_keyword(df_all: pd.DataFrame, per_kw_total: int, q_pos: int, q_neg: int, q_neu: int,
                               seed: int, col_tweet: str, col_keyword: str, col_stance: str, col_reason: str|None):
    """
    For each keyword, pick up to 'per_kw_total' rows with target quotas.
    Returns (selected_indices_set, dict: kw_slug -> [objects...])
    """
    # Build a working frame for selection
    work = df_all[[col_tweet, col_keyword, col_stance] + ([col_reason] if col_reason else [])].copy()
    work["_stance_norm"] = work[col_stance].map(normalize_stance)
    work["_idx"] = work.index
    # group by keyword value (original)
    selected_indices = set()
    payload_by_kw = {}

    # Unique keywords (stable order)
    keywords = work[col_keyword].fillna("").astype(str).tolist()
    # Preserve order of first appearance
    seen = set(); ordered_kws = []
    for k in keywords:
        if k not in seen:
            seen.add(k); ordered_kws.append(k)

    for kw in ordered_kws:
        sub = work[work[col_keyword] == kw]
        if sub.empty: continue

        kw_slug = re.sub(r"[^0-9a-zA-Z]+", "_", kw.strip().lower()).strip("_") or "unknown"
        rng = stable_rng(seed, kw_slug)

        # pools within keyword
        pos_pool = sub[sub["_stance_norm"] == "positive"]["_idx"].tolist()
        neg_pool = sub[sub["_stance_norm"] == "negative"]["_idx"].tolist()
        neu_pool = sub[sub["_stance_norm"] == "neutral"]["_idx"].tolist()
        other_pool = sub[~sub["_idx"].isin(pos_pool + neg_pool + neu_pool)]["_idx"].tolist()

        rng.shuffle(pos_pool); rng.shuffle(neg_pool); rng.shuffle(neu_pool); rng.shuffle(other_pool)

        chosen = []
        chosen += pos_pool[:min(q_pos, len(pos_pool))]
        chosen += neg_pool[:min(q_neg, len(neg_pool))]
        chosen += neu_pool[:min(q_neu, len(neu_pool))]

        # fill to per_kw_total from remaining within this keyword
        if len(chosen) < per_kw_total:
            remain = [i for i in (pos_pool + neg_pool + neu_pool + other_pool) if i not in chosen]
            rng.shuffle(remain)
            need = per_kw_total - len(chosen)
            chosen += remain[:need]

        # cap & record
        chosen = chosen[:per_kw_total]
        if not chosen:
            continue

        # Build JSON objects for this keyword
        items = []
        for idx in chosen:
            row = df_all.loc[idx]
            reason = clean_str(row[col_reason]) if col_reason else ""
            items.append({
                "entity": clean_str(row[col_keyword]),
                "statement": clean_str(row[col_tweet]),
                "stance": normalize_stance(row[col_stance]),
                "reason": reason
            })

        payload_by_kw[kw_slug] = items
        selected_indices.update(chosen)

    return selected_indices, payload_by_kw

# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=str, help="Folder containing many CSVs")
    ap.add_argument("--train-out", type=str, default="master_train.csv", help="Output master train CSV")
    ap.add_argument("--test-out", type=str, default="master_test.csv", help="Output master test CSV")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--label-col", type=str, default="stance", help="Label column for stratification (default 'stance')")
    # JSON knobs (per keyword)
    ap.add_argument("--json-outdir", type=str, default="jsons", help="Folder for per-keyword JSONs")
    ap.add_argument("--json-prefix", type=str, default="kyra", help="Filename prefix (e.g., 'kyra')")
    ap.add_argument("--json-per-keyword", type=int, default=10, help="Target items per keyword JSON (default 10)")
    ap.add_argument("--json-pos", type=int, default=3, help="Target positives per keyword (default 3)")
    ap.add_argument("--json-neg", type=int, default=3, help="Target negatives per keyword (default 3)")
    ap.add_argument("--json-neu", type=int, default=3, help="Target neutrals per keyword (default 3)")
    # Remainder split
    ap.add_argument("--test-frac", type=float, default=0.15, help="Test fraction on REMAINDER (default 0.15)")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[ERR] Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    csv_paths = sorted([p for p in input_dir.glob("*.csv") if p.is_file()])
    if not csv_paths:
        print(f"[ERR] No CSV files found in {input_dir}", file=sys.stderr); sys.exit(1)

    # Load all
    dfs, total_in = [], 0
    print(f"[INFO] Found {len(csv_paths)} CSVs. Loading...")
    for p in csv_paths:
        df = read_csv_robust(p)
        if df.empty:
            print(f"[WARN] {p.name}: empty/unreadable, skipping."); continue
        df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
        dfs.append(df); total_in += len(df)
        print(f"[OK] {p.name}: rows={len(df)}")

    if not dfs:
        print("[ERR] Nothing to process.", file=sys.stderr); sys.exit(1)

    all_df = concat_align(dfs)

    # Identify columns robustly
    col_tweet   = find_col(all_df, ["tweet", "text", "statement"])
    col_keyword = find_col(all_df, ["keyword", "entity", "topic"])
    col_stance  = find_col(all_df, ["stance", "label", "stance_label"])
    col_reason  = find_col(all_df, ["stance reason", "stance_reason", "reason"])

    if not (col_tweet and col_keyword and col_stance):
        print(f"[ERR] Required columns not found (tweet/keyword/stance). Found: "
              f"tweet={col_tweet}, keyword={col_keyword}, stance={col_stance}", file=sys.stderr)
        sys.exit(1)

    # ---------- Per-keyword JSON selection ----------
    sel_idx_set, payload_by_kw = pick_json_rows_per_keyword(
        all_df,
        per_kw_total=args.json_per_keyword,
        q_pos=args.json_pos, q_neg=args.json_neg, q_neu=args.json_neu,
        seed=args.seed,
        col_tweet=col_tweet, col_keyword=col_keyword, col_stance=col_stance, col_reason=col_reason
    )

    outdir = Path(args.json_outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    json_files_written = 0
    total_json_rows = 0
    for kw_slug, items in payload_by_kw.items():
        fname = f"{args.json_prefix}_{kw_slug}_stance.json"
        fpath = outdir / fname
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        json_files_written += 1
        total_json_rows += len(items)

    print(f"[INFO] Wrote {json_files_written} JSON files under {outdir} (total rows: {total_json_rows}).")

    # ---------- Split remainder (85:15) ----------
    remainder = all_df.drop(index=list(sel_idx_set)) if sel_idx_set else all_df.copy()

    try:
        df_train, df_test = stratified_split(remainder, args.test_frac, args.seed, args.label_col)
        split_mode = "stratified"
    except Exception:
        df_train, df_test = random_split(remainder, args.test_frac, args.seed)
        split_mode = "random"

    train_out = Path(args.train_out).expanduser().resolve()
    test_out  = Path(args.test_out).expanduser().resolve()
    df_train.to_csv(train_out, index=False, encoding="utf-8")
    df_test.to_csv(test_out, index=False, encoding="utf-8")

    # ---------- Summary ----------
    print("\n[SUMMARY]")
    print(f"  Files processed          : {len(csv_paths)}")
    print(f"  Total rows in            : {total_in}")
    print(f"  Keywords (JSON files)    : {json_files_written}")
    print(f"  JSON rows selected total : {total_json_rows}")
    print(f"  Remainder rows           : {len(remainder)}")
    print(f"  Split mode               : {split_mode}")
    print(f"  Train (85%)              : {len(df_train)} → {train_out}")
    print(f"  Test  (15%)              : {len(df_test)} → {test_out}")
    print("  Done.")

if __name__ == "__main__":
    main()
