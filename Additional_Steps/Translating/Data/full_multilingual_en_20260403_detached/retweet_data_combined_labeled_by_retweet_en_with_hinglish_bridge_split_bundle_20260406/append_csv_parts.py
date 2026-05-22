#!/usr/bin/env python3
"""Recombine split CSV parts into a single CSV file."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


csv.field_size_limit(sys.maxsize)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append split CSV parts back into a single CSV file."
    )
    parser.add_argument(
        "parts_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "parts",
        help="Directory containing the split CSV files (default: ./parts).",
    )
    parser.add_argument(
        "output_csv",
        nargs="?",
        type=Path,
        default=None,
        help="Output path for the recombined CSV. Defaults to a name derived from metadata.json.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding for input and output files (default: utf-8).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1_000_000,
        help="Print progress every N appended data rows (default: 1,000,000).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output CSV if it already exists.",
    )
    return parser.parse_args()


def default_output_path(bundle_dir: Path, parts_dir: Path) -> Path:
    metadata_path = bundle_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source_filename = metadata.get("source_filename")
        if source_filename:
            source_stem = Path(source_filename).stem
            return bundle_dir / f"{source_stem}_recombined.csv"
    return bundle_dir / f"{parts_dir.name}_recombined.csv"


def main() -> int:
    args = parse_args()
    parts_dir = args.parts_dir.resolve()
    bundle_dir = parts_dir.parent
    output_csv = args.output_csv.resolve() if args.output_csv else default_output_path(bundle_dir, parts_dir)

    if args.progress_every <= 0:
        raise SystemExit("--progress-every must be greater than zero.")

    if not parts_dir.exists():
        raise SystemExit(f"Parts directory does not exist: {parts_dir}")

    part_files = sorted(parts_dir.glob("*.csv"))
    if not part_files:
        raise SystemExit(f"No CSV part files found in {parts_dir}")

    if output_csv.exists() and not args.overwrite:
        raise SystemExit(
            f"Output CSV already exists: {output_csv}. Use --overwrite to replace it."
        )

    total_rows = 0
    expected_header = None

    with output_csv.open(
        "w",
        encoding=args.encoding,
        errors="surrogateescape",
        newline="",
    ) as output_handle:
        writer = csv.writer(output_handle)

        for index, part_file in enumerate(part_files, start=1):
            with part_file.open(
                "r",
                encoding=args.encoding,
                errors="surrogateescape",
                newline="",
            ) as input_handle:
                reader = csv.reader(input_handle)
                try:
                    header = next(reader)
                except StopIteration:
                    raise SystemExit(f"Encountered empty part file: {part_file}")

                if index == 1:
                    expected_header = header
                    writer.writerow(header)
                elif header != expected_header:
                    raise SystemExit(f"Header mismatch in part file: {part_file}")

                for row in reader:
                    writer.writerow(row)
                    total_rows += 1
                    if total_rows % args.progress_every == 0:
                        print(f"Appended {total_rows:,} data rows", flush=True)

            print(f"Merged {part_file.name}", flush=True)

    print(
        f"Finished merge: {len(part_files)} part files into {output_csv} "
        f"with {total_rows:,} data rows",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
