#!/usr/bin/env python3
"""
Merge English, native-script translation output, and Roman-script translation output
back into one CSV in original row order using the shared row_id column.
"""

import argparse
import csv
import heapq
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def normalize_english_row(row: Dict[str, str], text_col: str) -> Dict[str, str]:
    out = dict(row)
    out.setdefault("translation_engine", "unchanged")
    out.setdefault("translation_route", "english")
    out.setdefault("cosine_before_after", "")
    out["en_text"] = out.get("tweet_original") or out.get(text_col, "")
    return out


def next_row(reader: csv.DictReader, branch_name: str, text_col: str) -> Optional[Tuple[int, str, Dict[str, str]]]:
    try:
        row = next(reader)
    except StopIteration:
        return None
    if branch_name == "english":
        row = normalize_english_row(row, text_col=text_col)
    return (int(row["row_id"]), branch_name, row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--english", required=True)
    ap.add_argument("--native-translated", required=True)
    ap.add_argument("--roman-translated", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--text-col", default="tweet")
    args = ap.parse_args()

    input_specs = [
        ("english", Path(args.english)),
        ("native", Path(args.native_translated)),
        ("roman", Path(args.roman_translated)),
    ]

    readers: Dict[str, csv.DictReader] = {}
    handles = []
    all_fields: List[str] = []
    seen_fields = set()
    heap: List[Tuple[int, str, Dict[str, str]]] = []

    try:
        for branch_name, path in input_specs:
            handle = path.open("r", encoding="utf-8", newline="")
            handles.append(handle)
            reader = csv.DictReader(handle)
            readers[branch_name] = reader
            for field in reader.fieldnames or []:
                if field not in seen_fields:
                    seen_fields.add(field)
                    all_fields.append(field)
            item = next_row(reader, branch_name=branch_name, text_col=args.text_col)
            if item is not None:
                heapq.heappush(heap, item)

        for field in ("en_text", "translation_engine", "translation_route", "row_id"):
            if field not in seen_fields:
                all_fields.append(field)
                seen_fields.add(field)

        output_path = Path(args.output)
        output_path.unlink(missing_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as out_handle:
            writer = csv.DictWriter(out_handle, fieldnames=all_fields, extrasaction="ignore")
            writer.writeheader()
            written = 0
            while heap:
                _, branch_name, row = heapq.heappop(heap)
                writer.writerow({field: row.get(field, "") for field in all_fields})
                written += 1
                if written % 100_000 == 0:
                    print(f"[merge] wrote {written} rows")
                item = next_row(readers[branch_name], branch_name=branch_name, text_col=args.text_col)
                if item is not None:
                    heapq.heappush(heap, item)
        print(f"[done] merged output written to {output_path}")
    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
