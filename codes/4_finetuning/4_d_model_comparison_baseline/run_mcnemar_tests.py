#!/usr/bin/env python3
"""Run pairwise McNemar tests for baseline model-comparison result folders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
FINETUNING_DIR = THIS_FILE.parents[1]
sys.path.insert(0, str(FINETUNING_DIR))

from mcnemar_utils import (  # noqa: E402
    add_holm_correction,
    discover_prediction_files,
    pairwise_mcnemar,
    read_prediction_file,
    write_csv_rows,
    write_markdown_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute pairwise McNemar significance tests for existing prediction CSVs."
    )
    parser.add_argument(
        "--results-dir",
        action="append",
        type=Path,
        help=(
            "Result directory containing *_predictions.csv files. "
            "May be passed multiple times. Defaults to ./results and ./final_runs/results when present."
        ),
    )
    parser.add_argument(
        "--align-on",
        default="auto",
        help=(
            "Pair rows by 'auto', 'index', or one or more comma-separated CSV column names. "
            "Auto uses normalized tweet+keyword when available."
        ),
    )
    parser.add_argument("--output-prefix", default="mcnemar_tests")
    return parser.parse_args()


def default_result_dirs(base_dir: Path) -> list[Path]:
    candidates = [base_dir / "results", base_dir / "final_runs" / "results"]
    return [path for path in candidates if path.exists()]


def run_for_results_dir(results_dir: Path, align_on: str, output_prefix: str) -> None:
    prediction_files = discover_prediction_files(results_dir)
    if len(prediction_files) < 2:
        print(f"[skip] {results_dir}: fewer than two prediction files found")
        return

    predictions = []
    for model, path in prediction_files.items():
        loaded = read_prediction_file(path, model=model, align_on=align_on)
        predictions.append(loaded)
        print(
            f"[load] {results_dir.name}: {model} "
            f"valid={len(loaded.records)} total={loaded.total_rows} pred_col={loaded.pred_col}"
        )

    rows = pairwise_mcnemar(predictions)
    add_holm_correction(rows)

    csv_path = results_dir / f"{output_prefix}.csv"
    md_path = results_dir / f"{output_prefix}.md"
    write_csv_rows(csv_path, rows)
    write_markdown_table(md_path, f"McNemar Tests: {results_dir}", rows)
    print(f"[write] {csv_path}")
    print(f"[write] {md_path}")


def main() -> None:
    args = parse_args()
    base_dir = THIS_FILE.parent
    result_dirs = args.results_dir or default_result_dirs(base_dir)
    if not result_dirs:
        raise SystemExit("No result directories found. Pass --results-dir.")

    for results_dir in result_dirs:
        run_for_results_dir(results_dir.resolve(), args.align_on, args.output_prefix)


if __name__ == "__main__":
    main()
