#!/usr/bin/env python3
"""
Append explicit Hinglish audit columns to the merged English CSV.

For rows routed through the Roman/Hinglish branch, the added columns expose:
- the original Roman/Hinglish text
- the normalized Hindi/Devanagari text
- the English translation from that branch

All other rows receive empty values in those added columns.
"""

import argparse
import csv
from pathlib import Path


EXTRA_COLUMNS = [
    "hinglish_roman_text",
    "hinglish_hindi_text",
    "hinglish_english_text",
]


def load_roman_bridge(roman_translated_path: Path) -> dict[int, dict[str, str]]:
    bridge: dict[int, dict[str, str]] = {}
    with roman_translated_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_id = int(row["row_id"])
            bridge[row_id] = {
                "hinglish_roman_text": row.get("tweet_original", ""),
                "hinglish_hindi_text": row.get("tweet", ""),
                "hinglish_english_text": row.get("en_text", ""),
            }
    return bridge


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True, help="Merged English CSV")
    ap.add_argument("--roman-translated", required=True, help="Roman/Hinglish translated CSV")
    ap.add_argument("--output", required=True, help="Output CSV with appended Hinglish bridge columns")
    args = ap.parse_args()

    final_path = Path(args.final)
    roman_translated_path = Path(args.roman_translated)
    output_path = Path(args.output)

    bridge = load_roman_bridge(roman_translated_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)

    total_rows = 0
    roman_rows_in_final = 0
    missing_roman_bridge = 0

    with final_path.open("r", encoding="utf-8", newline="") as in_handle:
        reader = csv.DictReader(in_handle)
        fieldnames = list(reader.fieldnames or [])
        for col in EXTRA_COLUMNS:
            if col not in fieldnames:
                fieldnames.append(col)

        with output_path.open("w", encoding="utf-8", newline="") as out_handle:
            writer = csv.DictWriter(out_handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()

            for row in reader:
                total_rows += 1
                row_id = int(row["row_id"])
                extra = bridge.get(row_id)

                if row.get("translation_route") == "hinglish_roman":
                    roman_rows_in_final += 1
                    if extra is None:
                        missing_roman_bridge += 1

                for col in EXTRA_COLUMNS:
                    row[col] = ""

                if extra is not None:
                    row.update(extra)

                writer.writerow(row)

                if total_rows % 100_000 == 0:
                    print(f"[write] wrote {total_rows} rows")

    print(f"[done] output={output_path}")
    print(f"[summary] total_rows={total_rows}")
    print(f"[summary] roman_bridge_rows={len(bridge)}")
    print(f"[summary] roman_rows_in_final={roman_rows_in_final}")
    print(f"[summary] missing_roman_bridge={missing_roman_bridge}")


if __name__ == "__main__":
    main()
