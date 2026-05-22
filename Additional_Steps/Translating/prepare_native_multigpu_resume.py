#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd


def last_completed_total(log_path: Path) -> int:
    text = log_path.read_text(errors="ignore")
    matches = re.findall(r"chunk (\d+) processed (\d+) rows; total (\d+)", text)
    if not matches:
        raise SystemExit(f"No completed chunk totals found in {log_path}")
    return int(matches[-1][2])


def total_native_rows(route_log: Path) -> int:
    text = route_log.read_text(errors="ignore")
    matches = re.findall(r"\[chunk \d+\] rows=\d+ total=\d+ english=(\d+) native=(\d+) roman=(\d+)", text)
    if not matches:
        raise SystemExit(f"No routing totals found in {route_log}")
    total = 0
    for _, native, _ in matches:
        total += int(native)
    return total


def write_prefix(partial_path: Path, prefix_path: Path, prefix_rows: int, chunksize: int) -> None:
    if prefix_path.exists():
        prefix_path.unlink()
    wrote_header = False
    written = 0
    for chunk in pd.read_csv(partial_path, chunksize=chunksize):
        remaining = prefix_rows - written
        if remaining <= 0:
            break
        to_write = chunk.iloc[:remaining].copy()
        to_write.to_csv(prefix_path, mode="a", header=not wrote_header, index=False)
        wrote_header = True
        written += len(to_write)
        if written >= prefix_rows:
            break
    if written != prefix_rows:
        raise SystemExit(f"Expected to write {prefix_rows} prefix rows, wrote {written}")


def split_remaining(input_path: Path, output_dir: Path, skip_rows: int, num_shards: int, total_remaining: int, chunksize: int):
    for old in output_dir.glob("part_*.csv"):
        old.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_per_shard = math.ceil(total_remaining / max(num_shards, 1))
    shard_paths = [output_dir / f"part_{idx:02d}.csv" for idx in range(num_shards)]
    shard_counts = [0] * num_shards
    shard_idx = 0
    wrote_headers = [False] * num_shards
    remaining_to_skip = skip_rows
    rows_left = total_remaining

    for chunk in pd.read_csv(input_path, chunksize=chunksize):
        if remaining_to_skip >= len(chunk):
            remaining_to_skip -= len(chunk)
            continue
        if remaining_to_skip > 0:
            chunk = chunk.iloc[remaining_to_skip:].copy()
            remaining_to_skip = 0
        if rows_left <= 0:
            break
        start = 0
        while start < len(chunk) and shard_idx < num_shards and rows_left > 0:
            shard_limit = min(rows_per_shard, total_remaining - rows_per_shard * shard_idx)
            shard_remaining = shard_limit - shard_counts[shard_idx]
            if shard_remaining <= 0:
                shard_idx += 1
                continue
            end = min(start + shard_remaining, len(chunk))
            piece = chunk.iloc[start:end].copy()
            if piece.empty:
                break
            piece.to_csv(
                shard_paths[shard_idx],
                mode="a",
                header=not wrote_headers[shard_idx],
                index=False,
            )
            wrote_headers[shard_idx] = True
            shard_counts[shard_idx] += len(piece)
            rows_left -= len(piece)
            start = end
            if shard_counts[shard_idx] >= shard_limit:
                shard_idx += 1

    if rows_left != 0:
        raise SystemExit(f"Expected to distribute all remaining rows, rows_left={rows_left}")
    return shard_paths, shard_counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--chunksize", type=int, default=100_000)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    routed_dir = run_dir / "routed"
    translated_dir = run_dir / "translated"
    logs_dir = run_dir / "logs"
    work_dir = run_dir / "work" / "native_multigpu"
    shard_input_dir = work_dir / "inputs"
    manifest_path = work_dir / "manifest.json"

    input_path = routed_dir / "hindi_script_tweets.csv"
    partial_path = translated_dir / "hindi_script_en.csv"
    prefix_path = translated_dir / "hindi_script_en.prefix.csv"
    backup_partial_path = translated_dir / "hindi_script_en.interrupted_single_gpu.csv"

    route_log = logs_dir / "01_route.log"
    native_log = logs_dir / "02_translate_native.log"

    prefix_rows = last_completed_total(native_log)
    native_total = total_native_rows(route_log)
    remaining_rows = native_total - prefix_rows
    if remaining_rows <= 0:
        raise SystemExit("No remaining native rows to split")

    work_dir.mkdir(parents=True, exist_ok=True)
    write_prefix(partial_path, prefix_path, prefix_rows, args.chunksize)

    if backup_partial_path.exists():
        backup_partial_path.unlink()
    partial_path.rename(backup_partial_path)

    shard_paths, shard_counts = split_remaining(
        input_path=input_path,
        output_dir=shard_input_dir,
        skip_rows=prefix_rows,
        num_shards=args.num_shards,
        total_remaining=remaining_rows,
        chunksize=args.chunksize,
    )

    manifest = {
        "run_dir": str(run_dir),
        "prefix_rows": prefix_rows,
        "native_total_rows": native_total,
        "remaining_rows": remaining_rows,
        "prefix_path": str(prefix_path),
        "backup_partial_path": str(backup_partial_path),
        "shard_inputs": [
            {"path": str(path), "rows": rows}
            for path, rows in zip(shard_paths, shard_counts)
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_path)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
