#!/usr/bin/env python3
"""Run pairwise McNemar tests for multi-seed comparison outputs."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


THIS_FILE = Path(__file__).resolve()
FINETUNING_DIR = THIS_FILE.parents[1]
sys.path.insert(0, str(FINETUNING_DIR))

from mcnemar_utils import (  # noqa: E402
    add_holm_correction,
    pairwise_mcnemar,
    pooled_rows,
    read_prediction_file,
    write_csv_rows,
    write_markdown_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute pairwise McNemar significance tests for each seed in a multi-seed run."
    )
    parser.add_argument(
        "--run-root",
        action="append",
        type=Path,
        help=(
            "Run directory containing metrics_by_seed.csv. May be passed multiple times. "
            "Defaults to every runs/* directory with metrics_by_seed.csv."
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


def default_run_roots(base_dir: Path) -> List[Path]:
    runs_dir = base_dir / "runs"
    if not runs_dir.exists():
        return []
    return sorted(path for path in runs_dir.iterdir() if (path / "metrics_by_seed.csv").exists())


def load_prediction_manifest(run_root: Path) -> Dict[str, List[Tuple[str, Path]]]:
    metrics_path = run_root / "metrics_by_seed.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"{metrics_path} not found")

    by_seed: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)
    with metrics_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"seed", "model", "prediction_file"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{metrics_path} is missing required columns: {sorted(missing)}")
        for row in reader:
            pred_path = Path(row["prediction_file"])
            if not pred_path.is_absolute():
                pred_path = run_root / pred_path
            by_seed[str(row["seed"])].append((str(row["model"]), pred_path))
    return dict(by_seed)


def run_for_root(run_root: Path, align_on: str, output_prefix: str) -> None:
    manifest = load_prediction_manifest(run_root)
    by_seed_rows = []

    for seed, entries in sorted(manifest.items(), key=lambda item: int(item[0])):
        predictions = []
        print(f"[seed {seed}] loading {len(entries)} prediction files")
        for model, path in entries:
            loaded = read_prediction_file(path, model=model, align_on=align_on)
            predictions.append(loaded)
            print(
                f"  [load] {model} valid={len(loaded.records)} "
                f"total={loaded.total_rows} pred_col={loaded.pred_col}"
            )

        seed_rows = pairwise_mcnemar(predictions)
        for row in seed_rows:
            row["seed"] = seed
        by_seed_rows.extend(seed_rows)

    add_holm_correction(by_seed_rows, group_columns=("seed",))
    by_seed_csv = run_root / f"{output_prefix}_by_seed.csv"
    by_seed_md = run_root / f"{output_prefix}_by_seed.md"
    write_csv_rows(by_seed_csv, by_seed_rows)
    write_markdown_table(by_seed_md, f"McNemar Tests by Seed: {run_root}", by_seed_rows)
    print(f"[write] {by_seed_csv}")
    print(f"[write] {by_seed_md}")

    pooled = pooled_rows(by_seed_rows)
    add_holm_correction(pooled)
    pooled_csv = run_root / f"{output_prefix}_pooled.csv"
    pooled_md = run_root / f"{output_prefix}_pooled.md"
    write_csv_rows(pooled_csv, pooled)
    write_markdown_table(pooled_md, f"Pooled McNemar Tests: {run_root}", pooled)
    print(f"[write] {pooled_csv}")
    print(f"[write] {pooled_md}")


def main() -> None:
    args = parse_args()
    base_dir = THIS_FILE.parent
    run_roots = args.run_root or default_run_roots(base_dir)
    if not run_roots:
        raise SystemExit("No multi-seed run directories found. Pass --run-root.")

    for run_root in run_roots:
        run_for_root(run_root.resolve(), args.align_on, args.output_prefix)


if __name__ == "__main__":
    main()
