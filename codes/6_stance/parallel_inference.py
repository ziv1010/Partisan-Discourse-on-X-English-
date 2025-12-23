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
        --batch_size 32
"""

import os
import sys
import argparse
import subprocess
import tempfile
import shutil
import time
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


def run_worker(args_tuple):
    """Run a single worker on one GPU."""
    (chunk_file, gpu_id, worker_id, model_path, shots_dir, shots_prefix, 
     shots_json, output_dir, max_new_tokens, batch_size, 
     lora_adapter, merge_lora, trust_remote_code) = args_tuple
    
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
    
    # Set GPU for this worker
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"[Worker {worker_id}] Starting on GPU {gpu_id}, processing {chunk_file}")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd, 
            env=env, 
            capture_output=True, 
            text=True,
            check=True
        )
        elapsed = time.time() - start_time
        print(f"[Worker {worker_id}] Completed in {elapsed:.1f}s")
        return (worker_id, output_csv, True, None)
    except subprocess.CalledProcessError as e:
        print(f"[Worker {worker_id}] FAILED: {e.stderr[:500]}")
        return (worker_id, output_csv, False, e.stderr)


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
    parser.add_argument("--keep_temp", action="store_true", help="Keep temporary files")
    
    args = parser.parse_args()
    
    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    n_workers = len(gpu_ids)
    
    print(f"=" * 70)
    print(f"MULTI-GPU PARALLEL INFERENCE")
    print(f"=" * 70)
    print(f"Input: {args.input_csv}")
    print(f"Output: {args.output_csv}")
    print(f"Model: {args.model}")
    print(f"GPUs: {gpu_ids} ({n_workers} workers)")
    print(f"Batch size per worker: {args.batch_size}")
    print(f"=" * 70)
    
    # Create temp directory for intermediate files
    temp_dir = tempfile.mkdtemp(prefix="stance_parallel_")
    output_dir = os.path.join(temp_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Temp directory: {temp_dir}")
    
    try:
        # Split input CSV
        print(f"\nSplitting input CSV into {n_workers} chunks...")
        split_files = split_csv(args.input_csv, n_workers, temp_dir)
        print(f"Created {len(split_files)} chunks:")
        for chunk_file, idx, rows in split_files:
            print(f"  Chunk {idx}: {rows} rows")
        
        # Prepare worker arguments
        worker_args = []
        for (chunk_file, idx, _), gpu_id in zip(split_files, gpu_ids):
            worker_args.append((
                chunk_file, gpu_id, idx, args.model, args.shots_dir, 
                args.shots_prefix, args.shots_json, output_dir,
                args.max_new_tokens, args.batch_size,
                args.lora_adapter, args.merge_lora, args.trust_remote_code
            ))
        
        # Run workers in parallel
        print(f"\nStarting {len(worker_args)} parallel workers...")
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
                    print(f"[Worker {worker_id}] Exception: {e}")
                    results.append((worker_id, None, False, str(e)))
        
        elapsed = time.time() - start_time
        print(f"\nAll workers completed in {elapsed:.1f}s")
        
        # Check for failures
        failed = [r for r in results if not r[2]]
        if failed:
            print(f"\nWARNING: {len(failed)} worker(s) failed!")
            for worker_id, _, _, error in failed:
                print(f"  Worker {worker_id}: {error[:200] if error else 'Unknown error'}")
        
        # Merge results
        print(f"\nMerging results...")
        result_files = [r[1] for r in results if r[2] and r[1]]
        total_rows = merge_results(result_files, args.output_csv)
        
        print(f"\n" + "=" * 70)
        print(f"DONE!")
        print(f"Total rows processed: {total_rows}")
        print(f"Total time: {elapsed:.1f}s")
        print(f"Throughput: {total_rows / elapsed:.1f} rows/sec")
        print(f"Output: {args.output_csv}")
        print(f"=" * 70)
        
    finally:
        if not args.keep_temp:
            print(f"\nCleaning up temp directory...")
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            print(f"\nKeeping temp directory: {temp_dir}")


if __name__ == "__main__":
    main()
