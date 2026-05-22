#!/usr/bin/env python3
"""
Isolated temporal LoRA train/infer/evaluate runner.

This writes only under 4_f_temporal/runs/<run-name> by default.
It does not touch the existing lora_adaptor/ or final_runs/ outputs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
TEMPORAL_DIR = THIS_FILE.parent
FINETUNING_DIR = TEMPORAL_DIR.parent
PROJECT_ROOT = FINETUNING_DIR.parent.parent

TRAIN_CSV = TEMPORAL_DIR / "temporal_master_train_2020_2021.csv"
TEST_CSV = TEMPORAL_DIR / "temporal_master_test_2022_2023.csv"
MISTRAL_MODEL = Path("/scratch/ziv_baretto/Research_X/models/Mistral-7B-Instruct-v0.3")
FINETUNE_SCRIPT = FINETUNING_DIR / "4_a_DataProcessing" / "finetune.py"
INFER_SCRIPT = TEMPORAL_DIR / "infer_temporal_lora.py"
EVAL_SCRIPT = TEMPORAL_DIR / "evaluate_temporal_lora.py"
DEFAULT_SHOTS_DIR = FINETUNING_DIR / "4_a_DataProcessing" / "data_formatting" / "jsons"
DEFAULT_SHOTS_JSON = DEFAULT_SHOTS_DIR / "kyra_modi_stance.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=None, help="Defaults to 4_f_temporal/runs/temporal_lora_<timestamp>.")
    parser.add_argument("--run-name", default=None, help="Used only when --output-root is not supplied.")
    parser.add_argument("--resume", action="store_true", help="Allow reusing an existing output root.")
    parser.add_argument("--stage", choices=["all", "train", "infer", "eval"], default="all")
    parser.add_argument("--gpus", default="0", help="Comma-separated visible GPU ids. Training uses all; inference uses the first.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--infer-batch-size", type=int, default=8)
    parser.add_argument("--infer-max-input-length", type=int, default=2048,
                        help="Inference context length; defaults higher to keep few-shot examples.")
    parser.add_argument("--infer-max-new-tokens", type=int, default=96)
    parser.add_argument("--shots-dir", default=str(DEFAULT_SHOTS_DIR),
                        help="Directory with per-keyword few-shot JSONs like <prefix>_<keyword>_stance.json.")
    parser.add_argument("--shots-prefix", default="kyra",
                        help="Prefix for few-shot JSON filenames.")
    parser.add_argument("--shots-json", default=str(DEFAULT_SHOTS_JSON),
                        help="Fallback few-shot JSON when a per-keyword file is missing.")
    parser.add_argument("--no-fewshot", action="store_true",
                        help="Disable few-shot examples for temporal LoRA inference.")
    parser.add_argument("--model-dir", default=str(MISTRAL_MODEL))
    parser.add_argument("--train-csv", default=str(TRAIN_CSV))
    parser.add_argument("--test-csv", default=str(TEST_CSV))
    parser.add_argument("--dry-run", action="store_true", help="Print commands and write config, but do not execute.")
    return parser.parse_args()


def split_gpus(gpus: str) -> list[str]:
    return [g.strip() for g in str(gpus).split(",") if g.strip()]


def make_env(gpus: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def run_streamed(cmd: list[str], log_path: Path, env: dict[str, str], dry_run: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(shlex.quote(x) for x in cmd)
    print(f"\n[cmd] {printable}", flush=True)
    print(f"[log] {log_path}", flush=True)
    if dry_run:
        log_path.write_text(printable + "\n", encoding="utf-8")
        return

    with log_path.open("w", encoding="utf-8") as log:
        log.write(printable + "\n\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        code = proc.wait()
    if code:
        raise subprocess.CalledProcessError(code, cmd)


def build_paths(output_root: Path) -> dict[str, Path]:
    return {
        "adapter": output_root / "models" / "mistral_lora_adapter",
        "predictions": output_root / "predictions" / "temporal_mistral_lora_predictions.csv",
        "evaluation": output_root / "evaluation",
        "logs": output_root / "logs",
    }


def train_cmd(args: argparse.Namespace, adapter_dir: Path) -> list[str]:
    gpu_ids = split_gpus(args.gpus)
    if len(gpu_ids) > 1:
        launcher = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes",
            "1",
            "--nproc_per_node",
            str(len(gpu_ids)),
        ]
    else:
        launcher = [sys.executable]

    return launcher + [
        str(FINETUNE_SCRIPT),
        "--model-dir",
        str(Path(args.model_dir).expanduser().resolve()),
        "--data-csv",
        str(Path(args.train_csv).expanduser().resolve()),
        "--out-dir",
        str(adapter_dir),
        "--max-length",
        str(args.max_length),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--lr",
        str(args.lr),
        "--bf16",
        "--seed",
        str(args.seed),
    ]


def infer_cmd(args: argparse.Namespace, adapter_dir: Path, pred_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(INFER_SCRIPT),
        "--input-csv",
        str(Path(args.test_csv).expanduser().resolve()),
        "--model-dir",
        str(Path(args.model_dir).expanduser().resolve()),
        "--lora-adapter",
        str(adapter_dir),
        "--output-csv",
        str(pred_path),
        "--batch-size",
        str(args.infer_batch_size),
        "--max-input-length",
        str(args.infer_max_input_length),
        "--max-new-tokens",
        str(args.infer_max_new_tokens),
        "--bf16",
        "--resume",
    ]
    if args.no_fewshot:
        cmd.append("--no-fewshot")
    else:
        cmd.extend([
            "--shots-dir",
            str(Path(args.shots_dir).expanduser().resolve()),
            "--shots-prefix",
            args.shots_prefix,
            "--shots-json",
            str(Path(args.shots_json).expanduser().resolve()),
        ])
    return cmd


def eval_cmd(pred_path: Path, eval_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(EVAL_SCRIPT),
        "--pred-csv",
        str(pred_path),
        "--output-dir",
        str(eval_dir),
        "--gold-col",
        "stance",
        "--pred-col",
        "lora_stance",
    ]


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"temporal_lora_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else TEMPORAL_DIR / "runs" / run_name

    if output_root.exists() and not args.resume:
        raise FileExistsError(f"{output_root} already exists. Use --resume or choose a different --run-name/--output-root.")
    output_root.mkdir(parents=True, exist_ok=True)

    paths = build_paths(output_root)
    for p in paths.values():
        if p.suffix:
            p.parent.mkdir(parents=True, exist_ok=True)
        else:
            p.mkdir(parents=True, exist_ok=True)

    config = {
        "output_root": str(output_root),
        "train_csv": str(Path(args.train_csv).expanduser().resolve()),
        "test_csv": str(Path(args.test_csv).expanduser().resolve()),
        "model_dir": str(Path(args.model_dir).expanduser().resolve()),
        "adapter_dir": str(paths["adapter"]),
        "prediction_csv": str(paths["predictions"]),
        "evaluation_dir": str(paths["evaluation"]),
        "gpus": args.gpus,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "max_length": args.max_length,
        "infer_batch_size": args.infer_batch_size,
        "infer_max_input_length": args.infer_max_input_length,
        "shots_dir": str(Path(args.shots_dir).expanduser().resolve()),
        "shots_prefix": args.shots_prefix,
        "shots_json": str(Path(args.shots_json).expanduser().resolve()) if args.shots_json else "",
        "no_fewshot": args.no_fewshot,
        "stage": args.stage,
    }
    (output_root / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"[info] Temporal run root: {output_root}", flush=True)
    print("[info] This run writes only under the temporal run root.", flush=True)

    if args.stage in {"all", "train"}:
        adapter_ready = (paths["adapter"] / "adapter_model.safetensors").exists()
        if args.resume and adapter_ready:
            print(f"[info] Adapter already exists, skipping train: {paths['adapter']}", flush=True)
        else:
            run_streamed(train_cmd(args, paths["adapter"]), paths["logs"] / "train.log", make_env(args.gpus), args.dry_run)

    if args.stage in {"all", "infer"}:
        first_gpu = split_gpus(args.gpus)[0]
        run_streamed(infer_cmd(args, paths["adapter"], paths["predictions"]), paths["logs"] / "infer.log", make_env(first_gpu), args.dry_run)

    if args.stage in {"all", "eval"}:
        run_streamed(eval_cmd(paths["predictions"], paths["evaluation"]), paths["logs"] / "eval.log", os.environ.copy(), args.dry_run)

    print(f"\n[done] Outputs: {output_root}", flush=True)


if __name__ == "__main__":
    main()
