#!/usr/bin/env python3
"""Split a large CSV into smaller CSV parts while preserving row order."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


csv.field_size_limit(sys.maxsize)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a CSV into smaller CSV files with the header repeated in each part."
    )
    parser.add_argument("source_csv", type=Path, help="Path to the input CSV file.")
    parser.add_argument("output_dir", type=Path, help="Directory where split CSV parts will be written.")
    parser.add_argument(
        "--rows-per-part",
        type=int,
        default=1_000_000,
        help="Number of data rows per output file (default: 1,000,000).",
    )
    parser.add_argument(
        "--base-name",
        default=None,
        help="Base filename prefix for output parts. Defaults to the source filename stem.",
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
        help="Print progress every N data rows processed (default: 1,000,000).",
    )
    return parser.parse_args()


def ensure_output_dir_is_safe(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_csvs = sorted(output_dir.glob("*.csv"))
    if existing_csvs:
        joined = ", ".join(path.name for path in existing_csvs[:5])
        suffix = "" if len(existing_csvs) <= 5 else ", ..."
        raise SystemExit(
            f"Refusing to write into non-empty output directory {output_dir} "
            f"because it already contains CSV files: {joined}{suffix}"
        )


def part_path(output_dir: Path, base_name: str, part_number: int) -> Path:
    return output_dir / f"{base_name}_part_{part_number:04d}.csv"


def write_metadata(bundle_dir: Path, payload: dict) -> None:
    metadata_path = bundle_dir / "metadata.json"
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()

    source_csv = args.source_csv.resolve()
    output_dir = args.output_dir.resolve()
    bundle_dir = output_dir.parent
    base_name = args.base_name or source_csv.stem

    if args.rows_per_part <= 0:
        raise SystemExit("--rows-per-part must be greater than zero.")

    if args.progress_every <= 0:
        raise SystemExit("--progress-every must be greater than zero.")

    if not source_csv.exists():
        raise SystemExit(f"Source CSV does not exist: {source_csv}")

    ensure_output_dir_is_safe(output_dir)

    total_rows = 0
    part_number = 0
    rows_in_current_part = 0
    part_rows: list[dict[str, int | str]] = []
    current_output_handle = None
    writer = None
    header: list[str] | None = None

    def open_next_part() -> tuple[Path, object, csv._writer]:
        nonlocal part_number, rows_in_current_part
        part_number += 1
        rows_in_current_part = 0
        output_path = part_path(output_dir, base_name, part_number)
        handle = output_path.open(
            "w",
            encoding=args.encoding,
            errors="surrogateescape",
            newline="",
        )
        csv_writer = csv.writer(handle)
        csv_writer.writerow(header)
        part_rows.append({"file": output_path.name, "rows": 0})
        print(f"Started {output_path.name}", flush=True)
        return output_path, handle, csv_writer

    try:
        with source_csv.open(
            "r",
            encoding=args.encoding,
            errors="surrogateescape",
            newline="",
        ) as source_handle:
            reader = csv.reader(source_handle)

            try:
                header = next(reader)
            except StopIteration:
                raise SystemExit(f"Source CSV is empty: {source_csv}")

            _, current_output_handle, writer = open_next_part()

            for row in reader:
                if rows_in_current_part >= args.rows_per_part:
                    current_output_handle.close()
                    _, current_output_handle, writer = open_next_part()

                writer.writerow(row)
                rows_in_current_part += 1
                total_rows += 1
                part_rows[-1]["rows"] = rows_in_current_part

                if total_rows % args.progress_every == 0:
                    print(
                        f"Processed {total_rows:,} data rows across {part_number} part(s)",
                        flush=True,
                    )
    finally:
        if current_output_handle is not None and not current_output_handle.closed:
            current_output_handle.close()

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(source_csv),
        "source_filename": source_csv.name,
        "rows_per_part": args.rows_per_part,
        "data_rows": total_rows,
        "header_columns": len(header),
        "columns": header,
        "part_count": part_number,
        "parts_directory": output_dir.name,
        "parts": part_rows,
    }
    write_metadata(bundle_dir, metadata)

    print(
        f"Finished split: {total_rows:,} data rows into {part_number} part files",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
