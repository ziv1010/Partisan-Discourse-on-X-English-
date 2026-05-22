#!/usr/bin/env python3
"""
Multi-GPU Parallel Inference for the Temporal LoRA Model
=========================================================

Wraps infer_temporal_lora.py across multiple GPUs for large-scale inference.

Two parallelism modes (set with --gpus_per_worker):
  1 (default): one model instance per GPU — 8 GPUs → 8 workers.
               Best for this setup: Mistral-7B (~14 GB) fits easily on each
               49 GB RTX 6000 Ada, so no splitting is needed.
  2           : model split across 2 GPUs per worker via device_map="auto" —
               8 GPUs → 4 workers. Use only if VRAM is tight.

Usage (recommended — 8 independent workers):
    python parallel_infer_temporal_lora.py \
        --input_csv  TEMPORAL_SPLIT/split_2022_2023.csv \
        --model_dir  /path/to/Mistral-7B-Instruct-v0.3 \
        --lora_adapter /path/to/mistral_lora_adapter \
        --output_csv TEMPORAL_SPLIT/split_2022_2023_lora_labeled.csv \
        --gpus 0,1,2,3,4,5,6,7 \
        --batch_size 16 \
        --resume

Usage (model-parallel — 4 workers, each spanning 2 GPUs):
    python parallel_infer_temporal_lora.py ... --gpus_per_worker 2

Resume:
    Re-run the exact same command with --resume.  Each worker picks up where
    it left off from its partial result file inside the state directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# State / chunk helpers
# ---------------------------------------------------------------------------

def _state_dir_default(output_csv: str) -> Path:
    p = Path(output_csv).expanduser().resolve()
    return p.with_name(f"{p.stem}_parallel_state")


def _meta_path(state_dir: Path) -> Path:
    return state_dir / "state.json"


def _save_meta(state_dir: Path, input_csv: str, output_csv: str,
               gpu_groups: list[list[int]], chunks: list[tuple[Path, int, int]]) -> None:
    meta = {
        "input_csv": str(Path(input_csv).expanduser().resolve()),
        "output_csv": str(Path(output_csv).expanduser().resolve()),
        "gpu_groups": gpu_groups,
        "n_workers": len(gpu_groups),
        "chunks": [
            {"file": c[0].name, "worker_id": c[1], "rows": c[2]}
            for c in chunks
        ],
    }
    _meta_path(state_dir).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _load_meta(state_dir: Path) -> dict | None:
    p = _meta_path(state_dir)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _split_csv(input_csv: str, n_workers: int, state_dir: Path) -> list[tuple[Path, int, int]]:
    df = pd.read_csv(input_csv, low_memory=False)
    chunk_size = (len(df) + n_workers - 1) // n_workers
    chunks = []
    for i in range(n_workers):
        sl = df.iloc[i * chunk_size: (i + 1) * chunk_size]
        if sl.empty:
            break
        path = state_dir / f"chunk_{i}.csv"
        sl.to_csv(path, index=False)
        chunks.append((path, i, len(sl)))
    return chunks


def prepare_chunks(input_csv: str, output_csv: str,
                   gpu_groups: list[list[int]], state_dir: Path,
                   resume: bool) -> list[tuple[Path, int, int]]:
    input_resolved = str(Path(input_csv).expanduser().resolve())
    meta = _load_meta(state_dir) if resume else None

    if meta is not None:
        if meta.get("input_csv") != input_resolved:
            raise ValueError(
                f"Resume state was created for a different input CSV.\n"
                f"  stored : {meta['input_csv']}\n"
                f"  current: {input_resolved}"
            )
        if meta.get("n_workers") != len(gpu_groups):
            raise ValueError(
                f"Resume state has {meta['n_workers']} workers but "
                f"current run requests {len(gpu_groups)}."
            )
        chunks = []
        for c in meta["chunks"]:
            p = state_dir / c["file"]
            if not p.exists():
                raise FileNotFoundError(f"Missing chunk file for resume: {p}")
            chunks.append((p, int(c["worker_id"]), int(c["rows"])))
        print(f"  Reusing {len(chunks)} existing chunk(s) for resume.")
        return chunks

    chunks = _split_csv(input_csv, len(gpu_groups), state_dir)
    _save_meta(state_dir, input_csv, output_csv, gpu_groups, chunks)
    return chunks


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

WORKER_SCRIPT = Path(__file__).resolve().parent.parent.parent.parent / \
    "codes" / "4_finetuning" / "4_f_temporal" / "infer_temporal_lora.py"
FINETUNING_DIR = WORKER_SCRIPT.parent.parent
DEFAULT_SHOTS_DIR = FINETUNING_DIR / "4_a_DataProcessing" / "data_formatting" / "jsons"
DEFAULT_SHOTS_JSON = DEFAULT_SHOTS_DIR / "kyra_modi_stance.json"


def run_worker(args_tuple: tuple) -> tuple[int, str, bool, str | None]:
    (worker_id, chunk_file, gpu_ids, model_dir, lora_adapter,
     result_csv, log_file, batch_size, max_input_length,
     max_new_tokens, shots_dir, shots_prefix, shots_json,
     no_fewshot, bf16, trust_remote_code, resume) = args_tuple

    cmd = [
        sys.executable,
        str(WORKER_SCRIPT),
        "--input-csv",       str(chunk_file),
        "--model-dir",       model_dir,
        "--lora-adapter",    lora_adapter,
        "--output-csv",      str(result_csv),
        "--batch-size",      str(batch_size),
        "--max-input-length", str(max_input_length),
        "--max-new-tokens",  str(max_new_tokens),
    ]
    if no_fewshot:
        cmd.append("--no-fewshot")
    else:
        cmd.extend([
            "--shots-dir", str(shots_dir),
            "--shots-prefix", str(shots_prefix),
            "--shots-json", str(shots_json),
        ])
    if bf16:
        cmd.append("--bf16")
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    if resume:
        cmd.append("--resume")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    env["PYTHONUNBUFFERED"] = "1"

    gpu_label = ",".join(str(g) for g in gpu_ids)
    print(f"[Worker {worker_id}] GPU(s) {gpu_label} | chunk: {chunk_file.name} "
          f"| log: {log_file.name}", flush=True)

    start = time.time()
    try:
        with open(log_file, "w", encoding="utf-8") as lf:
            proc = subprocess.Popen(
                cmd, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            assert proc.stdout is not None
            tail: list[str] = []
            for raw in proc.stdout:
                line = raw.rstrip()
                lf.write(line + "\n")
                lf.flush()
                print(f"[Worker {worker_id}] {line}", flush=True)
                tail.append(line)
                if len(tail) > 200:
                    tail.pop(0)
            rc = proc.wait()

        elapsed = time.time() - start
        if rc == 0 or result_csv.exists():
            print(f"[Worker {worker_id}] Done in {elapsed:.1f}s", flush=True)
            return (worker_id, str(result_csv), True, None)
        err = "\n".join(tail[-30:]) if tail else f"exit code {rc}"
        print(f"[Worker {worker_id}] FAILED: {err[:400]}", flush=True)
        return (worker_id, str(result_csv), False, err)

    except Exception as exc:
        print(f"[Worker {worker_id}] EXCEPTION: {exc}", flush=True)
        return (worker_id, str(result_csv), False, str(exc))


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_results(result_files: list[str], output_csv: str) -> int:
    dfs = [pd.read_csv(f, low_memory=False) for f in result_files if Path(f).exists()]
    if not dfs:
        print("WARNING: no result files to merge.", flush=True)
        return 0
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(output_csv, index=False)
    print(f"Merged {len(dfs)} chunk(s) → {output_csv} ({len(merged):,} rows)", flush=True)
    return len(merged)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-GPU parallel wrapper for infer_temporal_lora.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_csv",   required=True,  help="Input CSV (tweet + keyword columns required)")
    parser.add_argument("--model_dir",   required=True,  help="Base model directory (Mistral-7B-Instruct-v0.3)")
    parser.add_argument("--lora_adapter",required=True,  help="LoRA adapter directory")
    parser.add_argument("--output_csv",  required=True,  help="Final merged output CSV")
    parser.add_argument("--gpus",        default="0,1,2,3,4,5,6,7",
                        help="Comma-separated GPU IDs to use")
    parser.add_argument("--gpus_per_worker", type=int, default=1, choices=[1, 2],
                        help="GPUs per worker. 1 = one model per GPU (faster on 49 GB cards). "
                             "2 = model split across 2 GPUs per worker via device_map=auto.")
    parser.add_argument("--batch_size",  type=int, default=16,
                        help="Batch size per worker. Can increase since GPUs have 49 GB VRAM.")
    parser.add_argument("--max_input_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens",   type=int, default=96)
    parser.add_argument("--shots_dir", default=str(DEFAULT_SHOTS_DIR),
                        help="Directory with per-keyword few-shot JSONs like <prefix>_<keyword>_stance.json")
    parser.add_argument("--shots_prefix", default="kyra",
                        help="Prefix for few-shot JSON filenames")
    parser.add_argument("--shots_json", default=str(DEFAULT_SHOTS_JSON),
                        help="Fallback few-shot JSON when a per-keyword file is missing")
    parser.add_argument("--no_fewshot", action="store_true",
                        help="Disable few-shot examples and use the original bare temporal prompt")
    parser.add_argument("--bf16",       action="store_true", default=True,
                        help="Use bf16 (recommended, default on)")
    parser.add_argument("--no_bf16",    action="store_true",
                        help="Disable bf16 (use fp16 instead)")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--resume",     action="store_true",
                        help="Resume from an interrupted run")
    parser.add_argument("--state_dir",  default=None,
                        help="State/chunk directory. Derived from output_csv if omitted.")
    parser.add_argument("--keep_state", action="store_true",
                        help="Keep state directory after completion (always kept on resume)")
    args = parser.parse_args()

    use_bf16 = args.bf16 and not args.no_bf16

    # --- Validate worker script exists ---
    if not WORKER_SCRIPT.exists():
        # Try a few fallback locations
        fallbacks = [
            Path(__file__).resolve().parent.parent.parent.parent /
            "codes" / "4_finetuning" / "4_f_temporal" / "infer_temporal_lora.py",
        ]
        for fb in fallbacks:
            if fb.exists():
                break
        else:
            raise FileNotFoundError(
                f"Cannot find infer_temporal_lora.py. Expected: {WORKER_SCRIPT}\n"
                "Pass the correct path or symlink it next to this script."
            )

    # --- GPU groups ---
    all_gpus = [int(g.strip()) for g in args.gpus.split(",")]
    gpw = args.gpus_per_worker
    if len(all_gpus) % gpw != 0:
        raise ValueError(
            f"Number of GPUs ({len(all_gpus)}) must be divisible by "
            f"--gpus_per_worker ({gpw})."
        )
    gpu_groups = [all_gpus[i: i + gpw] for i in range(0, len(all_gpus), gpw)]
    n_workers = len(gpu_groups)

    # --- State directory ---
    if args.state_dir:
        state_dir = Path(args.state_dir).expanduser().resolve()
    else:
        state_dir = _state_dir_default(args.output_csv)
    state_dir.mkdir(parents=True, exist_ok=True)
    output_dir = state_dir / "outputs"
    output_dir.mkdir(exist_ok=True)

    # --- Header ---
    sep = "=" * 70
    print(sep, flush=True)
    print("MULTI-GPU PARALLEL TEMPORAL LORA INFERENCE", flush=True)
    print(sep, flush=True)
    print(f"Input CSV    : {args.input_csv}", flush=True)
    print(f"Output CSV   : {args.output_csv}", flush=True)
    print(f"Model        : {args.model_dir}", flush=True)
    print(f"LoRA adapter : {args.lora_adapter}", flush=True)
    print(f"Few-shot     : {not args.no_fewshot}", flush=True)
    if not args.no_fewshot:
        print(f"Shots dir    : {args.shots_dir}", flush=True)
        print(f"Fallback JSON: {args.shots_json}", flush=True)
    print(f"GPUs         : {all_gpus}  ({n_workers} worker(s) × {gpw} GPU(s) each)", flush=True)
    print(f"Batch size   : {args.batch_size} per worker", flush=True)
    print(f"bf16         : {use_bf16}", flush=True)
    print(f"Resume       : {args.resume}", flush=True)
    print(f"State dir    : {state_dir}", flush=True)
    print(sep, flush=True)

    # --- Chunks ---
    chunks = prepare_chunks(
        args.input_csv, args.output_csv, gpu_groups, state_dir, args.resume
    )
    print(f"\nChunks ({len(chunks)}):", flush=True)
    for chunk_file, wid, nrows in chunks:
        print(f"  Worker {wid}: {nrows:,} rows — {chunk_file.name}", flush=True)

    # --- Build worker arg tuples ---
    worker_args = []
    for (chunk_file, worker_id, _), gpu_ids in zip(chunks, gpu_groups):
        result_csv = output_dir / f"result_{worker_id}.csv"
        log_file   = output_dir / f"worker_{worker_id}.log"
        worker_args.append((
            worker_id, chunk_file, gpu_ids,
            args.model_dir, args.lora_adapter,
            result_csv, log_file,
            args.batch_size, args.max_input_length, args.max_new_tokens,
            args.shots_dir, args.shots_prefix, args.shots_json,
            args.no_fewshot, use_bf16, args.trust_remote_code, args.resume,
        ))

    # --- Run ---
    print(f"\nStarting {n_workers} worker(s)...\n", flush=True)
    t0 = time.time()
    results: list[tuple[int, str, bool, str | None]] = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(run_worker, wa): wa[0] for wa in worker_args}
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as exc:
                wid = futures[fut]
                print(f"[Worker {wid}] Unhandled exception: {exc}", flush=True)
                results.append((wid, "", False, str(exc)))

    elapsed = time.time() - t0
    print(f"\nAll workers finished in {elapsed:.1f}s ({elapsed/3600:.2f}h)", flush=True)

    failed = [r for r in results if not r[2]]
    if failed:
        print(f"\nWARNING: {len(failed)} worker(s) failed:", flush=True)
        for wid, _, _, err in failed:
            print(f"  Worker {wid}: {(err or '')[:300]}", flush=True)

    # --- Merge ---
    print("\nMerging results...", flush=True)
    ok_files = [r[1] for r in results if r[2] and r[1]]
    total = merge_results(ok_files, args.output_csv)

    # --- Summary ---
    print(f"\n{sep}", flush=True)
    print("DONE", flush=True)
    print(f"{sep}", flush=True)
    print(f"Total rows written : {total:,}", flush=True)
    print(f"Total wall time    : {elapsed:.1f}s ({elapsed/3600:.2f}h)", flush=True)
    if total > 0 and elapsed > 0:
        print(f"Overall throughput : {total/elapsed:.1f} rows/sec", flush=True)
    print(f"Output             : {args.output_csv}", flush=True)
    print(f"State dir          : {state_dir}", flush=True)

    keep = args.keep_state or args.resume
    if not keep:
        print(f"\nCleaning up state dir: {state_dir}", flush=True)
        shutil.rmtree(state_dir, ignore_errors=True)
    else:
        print(f"\nState dir kept (use --resume to continue if interrupted).", flush=True)
    print(sep, flush=True)


if __name__ == "__main__":
    main()
