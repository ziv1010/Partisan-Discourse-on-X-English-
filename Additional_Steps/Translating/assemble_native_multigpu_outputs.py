#!/usr/bin/env python3
import argparse
import glob
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def count_rows(path: Path, chunksize: int) -> int:
    total = 0
    for chunk in pd.read_csv(path, chunksize=chunksize):
        total += len(chunk)
    return total


def concat_csvs(inputs: List[Path], output: Path, chunksize: int) -> None:
    if output.exists():
        output.unlink()
    wrote_header = False
    for path in inputs:
        for chunk in pd.read_csv(path, chunksize=chunksize):
            chunk.to_csv(output, mode="a", header=not wrote_header, index=False)
            wrote_header = True


def write_lowcos(input_path: Path, lowcos_path: Path, chunksize: int, inspect_quantile: float, highrisk_quantile: float) -> None:
    sims_all = []
    for chunk in pd.read_csv(input_path, chunksize=chunksize, usecols=["cosine_before_after"]):
        sims = chunk["cosine_before_after"].to_numpy(dtype=np.float32, copy=False)
        sims = sims[~np.isnan(sims)]
        sims_all.append(sims)
    if not sims_all:
        print("[lowcos] no cosine values found; skipping lowcos export")
        return

    sims_all_np = np.concatenate(sims_all)
    inspect_thr = float(np.quantile(sims_all_np, inspect_quantile))
    highrisk_thr = float(np.quantile(sims_all_np, highrisk_quantile))
    print(f"[lowcos] inspect_thr={inspect_thr:.4f} highrisk_thr={highrisk_thr:.4f}")

    if lowcos_path.exists():
        lowcos_path.unlink()
    wrote_header = False
    total_low = 0
    for chunk in pd.read_csv(input_path, chunksize=chunksize):
        low = chunk[chunk["cosine_before_after"] < inspect_thr].copy()
        if low.empty:
            continue
        low.to_csv(lowcos_path, mode="a", header=not wrote_header, index=False)
        wrote_header = True
        total_low += len(low)
    print(f"[lowcos] wrote {total_low} rows to {lowcos_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--shard-glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lowcos-out", required=True)
    ap.add_argument("--expected-total", type=int, required=True)
    ap.add_argument("--chunksize", type=int, default=100_000)
    ap.add_argument("--inspect-quantile", type=float, default=0.10)
    ap.add_argument("--highrisk-quantile", type=float, default=0.05)
    args = ap.parse_args()

    prefix = Path(args.prefix)
    shard_paths = [
        Path(path)
        for path in sorted(glob.glob(args.shard_glob))
        if not Path(path).name.endswith("_lowcos.csv")
    ]
    if not shard_paths:
        raise SystemExit(f"No shard outputs matched glob: {args.shard_glob}")

    inputs = [prefix] + shard_paths
    output = Path(args.out)
    lowcos_path = Path(args.lowcos_out)

    concat_csvs(inputs, output, args.chunksize)
    row_count = count_rows(output, args.chunksize)
    if row_count != args.expected_total:
        raise SystemExit(f"Expected {args.expected_total} rows after concat, found {row_count}")
    print(f"[concat] wrote {row_count} rows to {output}")

    write_lowcos(output, lowcos_path, args.chunksize, args.inspect_quantile, args.highrisk_quantile)


if __name__ == "__main__":
    main()
