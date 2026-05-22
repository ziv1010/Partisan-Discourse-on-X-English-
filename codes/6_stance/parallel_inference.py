#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-GPU Parallel Stance Inference

Runs multiple workers in parallel, each on a separate GPU, for ~4x throughput.
Each worker processes a chunk of the input CSV independently.

Usage:
    python parallel_inference.py \
        --input_csv /path/to/input.csv \
        --model /path/to/model \
        --shots_dir /path/to/shots \
        --output_csv /path/to/output.csv \
        --gpus 0,1,2,3 \
        --batch_size 32 \
        [--resume] [--state_dir /path/to/state_dir]
"""

import os
import sys
import argparse
import subprocess
import tempfile
import shutil
import time
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd


def split_csv(input_csv: str, n_parts: int, output_dir: str) -> list:
    """Split input CSV into n_parts files for parallel processing."""
    df = pd.read_csv(input_csv)
    total_rows = len(df)
    chunk_size = (total_rows + n_parts - 1) // n_parts
    
    split_files = []
    for i in range(n_parts):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_rows)
        if start_idx >= total_rows:
            break
        
        chunk_df = df.iloc[start_idx:end_idx]
        chunk_file = os.path.join(output_dir, f"chunk_{i}.csv")
        chunk_df.to_csv(chunk_file, index=False)
        split_files.append((chunk_file, i, len(chunk_df)))
    
    return split_files


def default_state_dir(output_csv: str) -> str:
    """Derive a stable working directory from the requested output path."""
    out_path = Path(output_csv).expanduser().resolve()
    return str(out_path.with_name(f"{out_path.stem}_parallel_state"))


def state_meta_path(state_dir: str) -> str:
    return os.path.join(state_dir, "state.json")


def save_state_metadata(state_dir: str, input_csv: str, output_csv: str, gpu_ids: list, split_files: list):
    """Persist enough metadata to safely resume an interrupted run."""
    meta = {
        "input_csv": str(Path(input_csv).expanduser().resolve()),
        "output_csv": str(Path(output_csv).expanduser().resolve()),
        "gpu_ids": list(gpu_ids),
        "n_workers": len(gpu_ids),
        "chunks": [
            {
                "file": os.path.basename(chunk_file),
                "worker_id": idx,
                "rows": rows,
            }
            for chunk_file, idx, rows in split_files
        ],
    }
    with open(state_meta_path(state_dir), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_state_metadata(state_dir: str):
    meta_file = state_meta_path(state_dir)
    if not os.path.exists(meta_file):
        return None
    with open(meta_file, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_split_files(input_csv: str, output_csv: str, gpu_ids: list, state_dir: str, resume: bool) -> list:
    """Reuse existing chunk files on resume; otherwise create them fresh."""
    input_csv_resolved = str(Path(input_csv).expanduser().resolve())
    meta = load_state_metadata(state_dir) if resume else None

    if meta is not None:
        if meta.get("input_csv") != input_csv_resolved:
            raise ValueError(
                "Resume state was created for a different input CSV.\n"
                f"  existing: {meta.get('input_csv')}\n"
                f"  current : {input_csv_resolved}"
            )
        if int(meta.get("n_workers", 0)) != len(gpu_ids):
            raise ValueError(
                "Resume state was created with a different worker count.\n"
                f"  existing: {meta.get('n_workers')}\n"
                f"  current : {len(gpu_ids)}"
            )

        split_files = []
        for chunk in meta.get("chunks", []):
            chunk_file = os.path.join(state_dir, chunk["file"])
            if not os.path.exists(chunk_file):
                raise FileNotFoundError(f"Missing chunk file required for resume: {chunk_file}")
            split_files.append((chunk_file, int(chunk["worker_id"]), int(chunk["rows"])))
        if split_files:
            return split_files

    split_files = split_csv(input_csv, len(gpu_ids), state_dir)
    save_state_metadata(state_dir, input_csv, output_csv, gpu_ids, split_files)
    return split_files


def run_worker(args_tuple):
    """Run a single worker on one GPU."""
    (chunk_file, gpu_id, worker_id, model_path, shots_dir, shots_prefix, 
     shots_json, output_dir, max_new_tokens, batch_size, 
     lora_adapter, merge_lora, trust_remote_code, resume) = args_tuple
    
    output_csv = os.path.join(output_dir, f"result_{worker_id}.csv")
    log_file = os.path.join(output_dir, f"worker_{worker_id}.log")
    
    # Build command
    cmd = [
        sys.executable, 
        os.path.join(os.path.dirname(__file__), "finetune_stance.py"),
        "--input_csv", chunk_file,
        "--model", model_path,
        "--shots_dir", shots_dir,
        "--shots_prefix", shots_prefix,
        "--output_csv", output_csv,
        "--max_new_tokens", str(max_new_tokens),
        "--batch_size", str(batch_size),
        "--bucket_by_length",
        "--save_every", "200",
        "--log_file", log_file,
    ]
    
    if shots_json:
        cmd.extend(["--shots_json", shots_json])
    if lora_adapter:
        cmd.extend(["--lora_adapter", lora_adapter])
    if merge_lora:
        cmd.append("--merge_lora")
    if trust_remote_code:
        cmd.append("--trust_remote_code")
    if resume:
        cmd.append("--resume")
    
    # Set GPU for this worker
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"[Worker {worker_id}] Starting on GPU {gpu_id}, processing {chunk_file}", flush=True)
    
    start_time = time.time()
    process = None
    streamed_lines = []
    try:
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if line:
                print(f"[Worker {worker_id}] {line}", flush=True)
                streamed_lines.append(line)
                if len(streamed_lines) > 200:
                    streamed_lines.pop(0)
        result_code = process.wait()
        elapsed = time.time() - start_time
        
        # Check if output file was created (actual success indicator)
        if result_code == 0 or os.path.exists(output_csv):
            print(f"[Worker {worker_id}] Completed in {elapsed:.1f}s", flush=True)
            return (worker_id, output_csv, True, None)
        else:
            error_msg = "\n".join(streamed_lines) if streamed_lines else f"Process exited with rc={result_code}"
            print(f"[Worker {worker_id}] FAILED (rc={result_code}): {error_msg[:500]}", flush=True)
            return (worker_id, output_csv, False, error_msg)
    except KeyboardInterrupt:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        raise
    except Exception as e:
        print(f"[Worker {worker_id}] EXCEPTION: {str(e)[:500]}", flush=True)
        return (worker_id, output_csv, False, str(e))



def merge_results(result_files: list, output_csv: str):
    """Merge all worker results into a single CSV."""
    dfs = []
    for f in result_files:
        if os.path.exists(f):
            dfs.append(pd.read_csv(f))
    
    if dfs:
        merged = pd.concat(dfs, axis=0, ignore_index=True)
        merged.to_csv(output_csv, index=False)
        print(f"Merged {len(dfs)} files -> {output_csv} ({len(merged)} rows)")
        return len(merged)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Parallel Stance Inference")
    parser.add_argument("--input_csv", required=True, help="Input CSV file")
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--shots_dir", required=True, help="Directory with few-shot JSONs")
    parser.add_argument("--shots_prefix", default="kyra", help="Prefix for shot files")
    parser.add_argument("--shots_json", default=None, help="Fallback shots JSON")
    parser.add_argument("--output_csv", required=True, help="Output CSV file")
    parser.add_argument("--gpus", default="0,1,2,3", help="Comma-separated GPU IDs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per worker")
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--lora_adapter", default=None)
    parser.add_argument("--merge_lora", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from an interrupted run using a stable state directory.")
    parser.add_argument("--state_dir", default=None, help="Directory for chunks, logs, and partial worker outputs. If omitted with --resume, a sidecar directory is derived from --output_csv.")
    parser.add_argument("--keep_temp", action="store_true", help="Keep temporary files")
    
    args = parser.parse_args()
    
    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    n_workers = len(gpu_ids)
    
    print(f"=" * 70, flush=True)
    print(f"MULTI-GPU PARALLEL INFERENCE", flush=True)
    print(f"=" * 70, flush=True)
    print(f"Input: {args.input_csv}", flush=True)
    print(f"Output: {args.output_csv}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"GPUs: {gpu_ids} ({n_workers} workers)", flush=True)
    print(f"Batch size per worker: {args.batch_size}", flush=True)
    print(f"=" * 70, flush=True)
    
    # Resume requires a stable working directory rather than a throwaway temp dir.
    if args.state_dir:
        state_dir = str(Path(args.state_dir).expanduser().resolve())
        state_dir_is_temp = False
        os.makedirs(state_dir, exist_ok=True)
    elif args.resume:
        state_dir = default_state_dir(args.output_csv)
        state_dir_is_temp = False
        os.makedirs(state_dir, exist_ok=True)
    else:
        state_dir = tempfile.mkdtemp(prefix="stance_parallel_")
        state_dir_is_temp = True

    output_dir = os.path.join(state_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"State directory: {state_dir}", flush=True)
    
    try:
        # Split input CSV or reuse the previous split for resume.
        print(f"\nPreparing {n_workers} chunk(s)...", flush=True)
        split_files = prepare_split_files(
            args.input_csv,
            args.output_csv,
            gpu_ids,
            state_dir,
            args.resume,
        )
        if args.resume:
            print(f"Prepared {len(split_files)} chunk(s) with resume support:", flush=True)
        else:
            print(f"Created {len(split_files)} chunk(s):", flush=True)
        for chunk_file, idx, rows in split_files:
            print(f"  Chunk {idx}: {rows} rows", flush=True)
        
        # Prepare worker arguments
        worker_args = []
        for (chunk_file, idx, _), gpu_id in zip(split_files, gpu_ids):
            worker_args.append((
                chunk_file, gpu_id, idx, args.model, args.shots_dir, 
                args.shots_prefix, args.shots_json, output_dir,
                args.max_new_tokens, args.batch_size,
                args.lora_adapter, args.merge_lora, args.trust_remote_code,
                args.resume,
            ))
        
        # Run workers in parallel
        print(f"\nStarting {len(worker_args)} parallel workers...", flush=True)
        start_time = time.time()
        
        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(run_worker, wa): wa[2] for wa in worker_args}
            for future in as_completed(futures):
                worker_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"[Worker {worker_id}] Exception: {e}", flush=True)
                    results.append((worker_id, None, False, str(e)))
        
        elapsed = time.time() - start_time
        print(f"\nAll workers completed in {elapsed:.1f}s", flush=True)
        
        # Check for failures
        failed = [r for r in results if not r[2]]
        if failed:
            print(f"\nWARNING: {len(failed)} worker(s) failed!", flush=True)
            for worker_id, _, _, error in failed:
                print(f"  Worker {worker_id}: {error[:200] if error else 'Unknown error'}", flush=True)
        
        # Merge results
        print(f"\nMerging results...", flush=True)
        result_files = [r[1] for r in results if r[2] and r[1]]
        total_rows = merge_results(result_files, args.output_csv)
        
        print(f"\n" + "=" * 70, flush=True)
        print(f"DONE!", flush=True)
        print(f"Total rows processed: {total_rows}", flush=True)
        print(f"Total time: {elapsed:.1f}s", flush=True)
        print(f"Throughput: {total_rows / elapsed:.1f} rows/sec", flush=True)
        print(f"Output: {args.output_csv}", flush=True)
        print(f"=" * 70, flush=True)
        
    finally:
        if state_dir_is_temp and not args.keep_temp:
            print(f"\nCleaning up state directory...", flush=True)
            shutil.rmtree(state_dir, ignore_errors=True)
        else:
            print(f"\nKeeping state directory: {state_dir}", flush=True)


if __name__ == "__main__":
    main()
