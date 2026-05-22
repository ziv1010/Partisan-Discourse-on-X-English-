#!/usr/bin/env python3
"""
Temporal Split: 2020-2021 vs 2022-2023

Reads the final stance CSV, joins timestamps from the bridge CSV on row_id,
then writes two separate CSVs — one per period — into this directory.
Prints overall and per-keyword stats. Does not modify any existing files.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent.resolve()

FINAL_CSV = (
    BASE_DIR.parent
    / "FINAL_STANCE"
    / "extracted_ALL_35keywords_english_hindi_only_final_with_existing_and_inferred_stance.csv"
)
BRIDGE_CSV = (
    BASE_DIR.parent
    / "SCRIPTS_OuTPUTS"
    / "retweet_data_combined_labeled_by_retweet_en_with_hinglish_bridge.csv"
)

OUT_EARLY = BASE_DIR / "split_2020_2021.csv"
OUT_LATE  = BASE_DIR / "split_2022_2023.csv"
OUT_STATS = BASE_DIR / "split_stats.txt"

EARLY_YEARS = {"2020", "2021"}
LATE_YEARS  = {"2022", "2023"}


def build_year_lookup(target_ids: set[str]) -> pd.DataFrame:
    print("Building year lookup from bridge CSV (chunked)...")
    chunks = []
    for chunk in pd.read_csv(
        BRIDGE_CSV,
        usecols=["row_id", "timestamp"],
        dtype="string",
        chunksize=250_000,
        low_memory=False,
    ):
        chunk["row_id"] = chunk["row_id"].fillna("").str.strip()
        chunk["timestamp"] = chunk["timestamp"].fillna("").str.strip()
        matched = chunk[chunk["row_id"].isin(target_ids)].copy()
        if not matched.empty:
            matched["year"] = matched["timestamp"].str.slice(0, 4)
            chunks.append(matched[["row_id", "year"]])

    lookup = (
        pd.concat(chunks, ignore_index=True)
        .drop_duplicates(subset=["row_id"], keep="first")
    )
    print(f"  Matched {len(lookup):,} unique row_ids with timestamps")
    return lookup


def print_and_save_stats(df_early: pd.DataFrame, df_late: pd.DataFrame, lines: list[str]) -> None:
    def section(title: str, df: pd.DataFrame) -> None:
        lines.append(f"\n{'='*60}")
        lines.append(title)
        lines.append(f"{'='*60}")
        lines.append(f"  Total rows : {len(df):,}")
        lines.append(f"  Unique tweet-ids : {df['row_id'].nunique():,}")
        lines.append("")
        lines.append(f"  {'Keyword':<25} {'Rows':>8}  {'Pro Ruling':>12}  {'Pro Opp':>10}")
        lines.append(f"  {'-'*25} {'-'*8}  {'-'*12}  {'-'*10}")
        for kw, grp in df.groupby("keyword", sort=True):
            ruling = (grp["_label_norm"] == "pro ruling").sum()
            opp    = (grp["_label_norm"] == "pro opposition").sum()
            lines.append(f"  {kw:<25} {len(grp):>8,}  {ruling:>12,}  {opp:>10,}")

    section("PERIOD: 2020–2021", df_early)
    section("PERIOD: 2022–2023", df_late)

    lines.append(f"\n{'='*60}")
    lines.append("COMBINED TOTALS")
    lines.append(f"{'='*60}")
    lines.append(f"  2020–2021 rows : {len(df_early):,}")
    lines.append(f"  2022–2023 rows : {len(df_late):,}")
    lines.append(f"  Grand total    : {len(df_early) + len(df_late):,}")

    for line in lines:
        print(line)


def main() -> None:
    print(f"Loading final stance CSV: {FINAL_CSV}")
    df = pd.read_csv(FINAL_CSV, dtype="string", low_memory=False)
    df["row_id"] = df["row_id"].fillna("").str.strip()
    print(f"  Loaded {len(df):,} rows")

    target_ids = set(df["row_id"].unique()) - {""}
    year_lookup = build_year_lookup(target_ids)

    df = df.merge(year_lookup, on="row_id", how="left")
    missing = df["year"].isna().sum()
    if missing:
        print(f"  WARNING: {missing:,} rows have no year after join — they will be excluded from both splits")

    df_early = df[df["year"].isin(EARLY_YEARS)].drop(columns=["year"]).reset_index(drop=True)
    df_late  = df[df["year"].isin(LATE_YEARS)].drop(columns=["year"]).reset_index(drop=True)

    df_early.to_csv(OUT_EARLY, index=False)
    df_late.to_csv(OUT_LATE, index=False)
    print(f"\nSaved: {OUT_EARLY}")
    print(f"Saved: {OUT_LATE}")

    lines: list[str] = ["TEMPORAL SPLIT STATS", f"Final CSV: {FINAL_CSV}"]
    print_and_save_stats(df_early, df_late, lines)

    OUT_STATS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nStats saved to: {OUT_STATS}")


if __name__ == "__main__":
    main()
