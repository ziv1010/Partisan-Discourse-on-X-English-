#!/usr/bin/env python3
"""
label_from_combined.py
──────────────────────
• Reads a *combined* tweets CSV in CHUNKSIZE batches
• Labels each row by matching retweet_author → party (from an Excel mapping)
• Keeps original_author column intact
• Writes:
      <out>.csv  (all columns + retweet_party [+ optional original_party])
      (+ optional unmatched_retweet.csv, unmatched_original.csv)
• Prints exact row & match stats
"""

import argparse, os, time
from typing import Tuple, Set, Dict
import pandas as pd
from tqdm import tqdm

# ───── defaults (can be overridden by CLI) ─────
CHUNKSIZE_DEFAULT = 100_000


def timed(msg: str):
    class _Timer:
        def __enter__(self):
            self.t0 = time.time(); print(msg)
        def __exit__(self, *exc):
            print(f"      done in {time.time() - self.t0:.1f}s\n")
    return _Timer()


def load_mapping(mapping_path: str) -> pd.DataFrame:
    """
    Expect an Excel with columns: politician, party
    Dedup by lowercase handle.
    """
    with timed("[1/4] Loading party mapping"):
        mp = pd.read_excel(mapping_path, sheet_name=0,
                           dtype={"politician": "string", "party": "string"})
        mp["author_lc"] = mp["politician"].astype("string").str.strip().str.lower()
        mp = (
            mp[["author_lc", "party"]]
            .dropna(subset=["author_lc"])
            .drop_duplicates("author_lc", keep="first")
            .reset_index(drop=True)
        )
        print(f"      Unique handles in mapping: {len(mp):,}")
        return mp


def prepare_outputs(out_path: str):
    """Remove existing output if present."""
    if os.path.exists(out_path):
        os.remove(out_path)


def process_chunk(
    chunk: pd.DataFrame,
    mapping_df: pd.DataFrame,
    header_written: Dict[str, bool],
    out_path: str,
    add_original_party: bool
) -> Tuple[int, int, int, Set[str], Set[str]]:
    """
    Merge retweet_author → retweet_party (required).
    Optionally also original_author → original_party (if add_original_party=True).
    Write appended CSV.
    Return: total_rows, matched_retweet, matched_original, unmatched_retweet_set, unmatched_original_set
    """
    # Normalize handles
    chunk["retweet_lc"]  = chunk["retweet_author"].astype("string").str.strip().str.lower()
    chunk["original_lc"] = chunk["original_author"].astype("string").str.strip().str.lower()

    # Merge for retweet party
    lab = chunk.merge(mapping_df, left_on="retweet_lc", right_on="author_lc", how="left")
    lab = lab.rename(columns={"party": "retweet_party"}).drop(columns=["author_lc"])

    # Optional: also map original_author → original_party
    unmatched_original_set: Set[str] = set()
    matched_original = 0
    if add_original_party:
        lab = lab.merge(mapping_df, left_on="original_lc", right_on="author_lc", how="left")
        lab = lab.rename(columns={"party": "original_party"}).drop(columns=["author_lc"])
        matched_original = int(lab["original_party"].notna().sum())
        unmatched_original_set = set(
            lab.loc[lab["original_party"].isna(), "original_author"].dropna().unique()
        )

    # Track unmatched retweet handles
    unmatched_retweet_set = set(
        lab.loc[lab["retweet_party"].isna(), "retweet_author"].dropna().unique()
    )
    matched_retweet = int(lab["retweet_party"].notna().sum())

    # Write out (append). Keep every row; party columns may be NaN for unmatched.
    # This preserves full combined data + new columns.
    cols_order = list(chunk.columns)  # original columns first
    # Ensure the new party columns appear at the end in a stable order
    for extra in (["retweet_party"] + (["original_party"] if add_original_party else [])):
        if extra not in cols_order:
            cols_order.append(extra)

    lab.to_csv(out_path, mode="a", index=False, header=not header_written["out"], columns=cols_order)
    header_written["out"] = True

    return len(chunk), matched_retweet, matched_original, unmatched_retweet_set, unmatched_original_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined", required=True,
                    help="Path to combined CSV (e.g., retweet_data_combined.csv)")
    ap.add_argument("--mapping", required=True,
                    help="Excel with columns: politician, party (for handle→party mapping)")
    ap.add_argument("--out", default=None,
                    help="Output CSV path (default: <combined_basename>_labeled_by_retweet.csv)")
    ap.add_argument("--unmatched_dir", default=None,
                    help="If set, writes unmatched_retweet.csv (and unmatched_original.csv if enabled)")
    ap.add_argument("--chunksize", type=int, default=CHUNKSIZE_DEFAULT,
                    help=f"Chunk size for streaming read (default: {CHUNKSIZE_DEFAULT})")
    ap.add_argument("--add-original-party", action="store_true",
                    help="Also map original_author → original_party")
    args = ap.parse_args()

    combined_path = args.combined
    mapping_path  = args.mapping
    out_path = args.out or os.path.splitext(combined_path)[0] + "_labeled_by_retweet.csv"

    # Prep
    mapping_df = load_mapping(mapping_path)
    prepare_outputs(out_path)

    # Stats
    total_rows = matched_retweet_total = matched_original_total = 0
    unmatched_retweet_all: Set[str] = set()
    unmatched_original_all: Set[str] = set()
    header_written = {"out": False}

    # Stream combined CSV
    with timed(f"[2/4] Streaming combined CSV → {os.path.basename(out_path)}"):
        for chunk in tqdm(pd.read_csv(combined_path, dtype="string", engine="c",
                                      chunksize=args.chunksize), unit="chunk"):
            n_rows, m_rtw, m_org, un_rtw, un_org = process_chunk(
                chunk, mapping_df, header_written, out_path, args.add_original_party
            )
            total_rows += n_rows
            matched_retweet_total  += m_rtw
            matched_original_total += m_org
            unmatched_retweet_all.update(un_rtw)
            unmatched_original_all.update(un_org)

    # Write unmatched lists (optional)
    if args.unmatched_dir:
        with timed("[3/4] Writing unmatched handle lists"):
            os.makedirs(args.unmatched_dir, exist_ok=True)
            pd.DataFrame(sorted(unmatched_retweet_all), columns=["retweet_author"]) \
              .to_csv(os.path.join(args.unmatched_dir, "unmatched_retweet.csv"), index=False)
            if args.add_original_party:
                pd.DataFrame(sorted(unmatched_original_all), columns=["original_author"]) \
                  .to_csv(os.path.join(args.unmatched_dir, "unmatched_original.csv"), index=False)

    # Summary
    print("\n[4/4] Summary")
    print("─────────────")
    print(f"Total rows processed        : {total_rows:,}")
    print(f"Rows with retweet_party     : {matched_retweet_total:,} "
          f"({matched_retweet_total/total_rows*100:,.2f}% of all rows)")
    if args.add_original_party:
        print(f"Rows with original_party    : {matched_original_total:,} "
              f"({matched_original_total/total_rows*100:,.2f}% of all rows)")
    if args.unmatched_dir:
        print(f"Unmatched retweet handles   : {len(unmatched_retweet_all):,} → "
              f"{os.path.join(args.unmatched_dir, 'unmatched_retweet.csv')}")
        if args.add_original_party:
            print(f"Unmatched original handles  : {len(unmatched_original_all):,} → "
                  f"{os.path.join(args.unmatched_dir, 'unmatched_original.csv')}")
    print(f"\n✅ Labeled CSV written to: {out_path}")


if __name__ == "__main__":
    main()
