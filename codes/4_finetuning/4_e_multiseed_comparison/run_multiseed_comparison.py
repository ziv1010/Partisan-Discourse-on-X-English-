#!/usr/bin/env python3
"""
Run the fixed-split, multi-seed stance model comparison.

This runner creates normalized 3-class train/dev/test CSVs once, then fine-tunes
each model from its base checkpoint for each requested seed. Outputs are written
under a new run directory, so existing model artifacts, results, and the old
Mistral LoRA adapter are not overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


THIS_FILE = Path(__file__).resolve()
MULTISEED_DIR = THIS_FILE.parent
FINETUNING_DIR = MULTISEED_DIR.parent
PROJECT_ROOT = FINETUNING_DIR.parent.parent

DATA_DIR = FINETUNING_DIR / "4_a_DataProcessing" / "data_formatting"
MASTER_TRAIN = DATA_DIR / "master_train.csv"
MASTER_TEST = DATA_DIR / "master_test.csv"
SHOTS_DIR = DATA_DIR / "jsons"
SHOTS_JSON = SHOTS_DIR / "kyra_modi_stance.json"

BERT_SCRIPT = FINETUNING_DIR / "4_d_model_comparison_baseline" / "final_runs" / "scripts" / "bert_stance.py"
ROBERTA_SCRIPT = FINETUNING_DIR / "4_d_model_comparison_baseline" / "final_runs" / "scripts" / "roberta_stance.py"
PYABSA_SCRIPT = FINETUNING_DIR / "4_d_model_comparison_baseline" / "final_runs" / "scripts" / "pyabsa_stance.py"
MISTRAL_BASE_SCRIPT = FINETUNING_DIR / "4_d_model_comparison_baseline" / "final_runs" / "scripts" / "mistral_base_inference.py"
MISTRAL_TRAIN_SCRIPT = FINETUNING_DIR / "4_a_DataProcessing" / "finetune.py"
MISTRAL_INFER_SCRIPT = FINETUNING_DIR / "4_b_TestingComparison" / "finetune_stance.py"
MISTRAL_MODEL = Path("/scratch/ziv_baretto/Research_X/models/Mistral-7B-Instruct-v0.3")

LABELS = ["For", "Against", "Neutral"]
PREDICTION_CONFIG = {
    "bert": {"file": "bert_predictions.csv", "pred_col": "bert_prediction"},
    "roberta": {"file": "roberta_predictions.csv", "pred_col": "roberta_prediction"},
    "pyabsa": {"file": "pyabsa_predictions.csv", "pred_col": "pyabsa_prediction"},
    "mistral_base": {"file": "mistral_base_predictions.csv", "pred_col": "mistral_prediction"},
    "mistral_fewshot": {"file": "mistral_fewshot_predictions.csv", "pred_col": "fewshot_label"},
    "mistral_lora": {"file": "mistral_lora_predictions.csv", "pred_col": "fewshot_label"},
}

# Models whose predictions are fully deterministic (greedy decode, no random
# shot subsampling beyond what the seed already controls). We run them once
# per output_root and copy the predictions CSV to other seed dirs to save
# wall-clock time across seeds.
DETERMINISTIC_INFERENCE_MODELS = {"mistral_base", "mistral_fewshot"}

ENV_PACKAGES = [
    "numpy==1.26.4",
    "pandas==2.1.4",
    "scikit-learn",
    "tqdm",
    "matplotlib",
    "seaborn",
    "torch==2.5.1",
    "transformers==4.56.0",
    "accelerate>=1.3.0,<2.0",
    "datasets>=2.20.0",
    "peft>=0.13.2",
    "trl>=0.11.4",
    "bitsandbytes>=0.43.0",
    "sentencepiece",
    "protobuf",
    "langchain==1.1.3",
    "langchain-core==1.1.3",
    "pyabsa==2.4.1.post1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed stance model comparison.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 100])
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["bert", "roberta", "pyabsa", "mistral_base", "mistral_fewshot", "mistral_lora"],
        default=["bert", "roberta", "pyabsa", "mistral_base", "mistral_fewshot", "mistral_lora"],
    )
    parser.add_argument("--gpus", default="0,1,2,3", help="Visible GPU ids for all subprocesses.")
    parser.add_argument("--output-root", default=None, help="Run directory. Defaults to timestamped runs/run_*.") 
    parser.add_argument("--resume", action="store_true", help="Skip model runs whose prediction files already exist.")
    parser.add_argument("--prepare-only", action="store_true", help="Only create normalized fixed splits and config files.")
    parser.add_argument(
        "--parallel-models",
        action="store_true",
        help="Run models for a seed concurrently, assigning GPUs round-robin from --gpus.",
    )
    parser.add_argument("--split-seed", type=int, default=42, help="Fixed train/dev split seed.")
    parser.add_argument("--dev-size", type=float, default=0.10, help="Fixed dev fraction split from master_train.csv.")

    parser.add_argument("--bert-epochs", type=int, default=3)
    parser.add_argument("--bert-batch-size", type=int, default=16)
    parser.add_argument("--roberta-epochs", type=int, default=3)
    parser.add_argument("--roberta-batch-size", type=int, default=16)
    parser.add_argument("--pyabsa-epochs", type=int, default=5)
    parser.add_argument("--pyabsa-batch-size", type=int, default=16)
    parser.add_argument("--mistral-epochs", type=int, default=3)
    parser.add_argument("--mistral-batch-size", type=int, default=2)
    parser.add_argument("--mistral-grad-accum", type=int, default=8)
    parser.add_argument("--mistral-lr", type=float, default=2e-4)
    parser.add_argument("--mistral-max-length", type=int, default=1024)
    parser.add_argument("--mistral-infer-batch-size", type=int, default=16)

    parser.add_argument("--env-name", default="stance_multiseed")
    parser.add_argument("--create-env", action="store_true", help="Create/update a micromamba env, then re-run inside it.")
    parser.add_argument("--no-reexec", action="store_true", help="Do not re-run this script inside --env-name.")
    return parser.parse_args()


def run_checked(cmd: List[str], env: Optional[Dict[str, str]] = None) -> None:
    print("+ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def micromamba_path() -> str:
    mm = shutil.which("micromamba") or "/usr/local/bin/micromamba"
    if not Path(mm).exists():
        raise FileNotFoundError("micromamba was not found on PATH or at /usr/local/bin/micromamba")
    return mm


def env_exists(env_name: str) -> bool:
    mm = micromamba_path()
    proc = subprocess.run([mm, "env", "list"], check=True, capture_output=True, text=True)
    return any(line.split() and line.split()[0] == env_name for line in proc.stdout.splitlines())


def ensure_env(env_name: str) -> None:
    mm = micromamba_path()
    if not env_exists(env_name):
        run_checked([mm, "create", "-y", "-n", env_name, "python=3.11", "pip"])
    run_checked([mm, "run", "-n", env_name, "python", "-m", "pip", "install", "--upgrade", "pip"])
    run_checked([mm, "run", "-n", env_name, "python", "-m", "pip", "install", *ENV_PACKAGES])


def reexec_in_env(args: argparse.Namespace) -> None:
    if args.no_reexec or os.environ.get("MULTISEED_RUNNER_IN_ENV") == "1":
        return
    mm = micromamba_path()
    argv = [a for a in sys.argv[1:] if a != "--create-env"]
    env = os.environ.copy()
    env["MULTISEED_RUNNER_IN_ENV"] = "1"
    cmd = [mm, "run", "-n", args.env_name, "python", str(THIS_FILE), *argv]
    print("Re-running inside micromamba env:", " ".join(shlex.quote(c) for c in cmd), flush=True)
    os.execvpe(mm, cmd, env)


def normalize_stance(value) -> Optional[str]:
    if value is None:
        return None
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except Exception:
        pass
    s = str(value).strip().lower()
    if not s:
        return None
    s = s.replace(".", "")
    s = s.replace("favour", "for").replace("favor", "for").replace("in_favor", "for")
    s = s.replace("in favour", "for").replace("in favor", "for")
    s = s.replace("nuetral", "neutral")
    mapping = {
        "for": "For",
        "positive": "For",
        "support": "For",
        "supports": "For",
        "against": "Against",
        "negative": "Against",
        "oppose": "Against",
        "opposes": "Against",
        "opposed": "Against",
        "neutral": "Neutral",
        "unrelated": "Neutral",
        "irrelevant": "Neutral",
    }
    return mapping.get(s)


def normalize_frame(df, split_name: str):
    before = len(df)
    df = df.copy()
    df["stance"] = df["stance"].apply(normalize_stance)
    df = df[df["stance"].isin(LABELS)].reset_index(drop=True)
    dropped = before - len(df)
    print(f"{split_name}: kept {len(df)} rows, dropped {dropped} blank/unsupported labels")
    return df


def prepare_splits(args: argparse.Namespace, output_root: Path) -> Dict[str, Path]:
    import pandas as pd
    from sklearn.model_selection import train_test_split

    split_dir = output_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    train_path = split_dir / "train.csv"
    dev_path = split_dir / "dev.csv"
    test_path = split_dir / "test.csv"
    manifest_path = split_dir / "split_manifest.json"

    if args.resume and train_path.exists() and dev_path.exists() and test_path.exists():
        return {"train": train_path, "dev": dev_path, "test": test_path}

    master_train = normalize_frame(pd.read_csv(MASTER_TRAIN), "master_train")
    master_test = normalize_frame(pd.read_csv(MASTER_TEST), "master_test")

    train_df, dev_df = train_test_split(
        master_train,
        test_size=args.dev_size,
        random_state=args.split_seed,
        stratify=master_train["stance"],
    )
    train_df = train_df.reset_index(drop=True)
    dev_df = dev_df.reset_index(drop=True)

    for name, df in [("train", train_df), ("dev", dev_df), ("test", master_test)]:
        bad = sorted(set(df["stance"]) - set(LABELS))
        if bad:
            raise ValueError(f"{name} split contains non-3-class labels: {bad}")

    train_df.to_csv(train_path, index=False)
    dev_df.to_csv(dev_path, index=False)
    master_test.to_csv(test_path, index=False)

    manifest = {
        "source_train": str(MASTER_TRAIN),
        "source_test": str(MASTER_TEST),
        "split_seed": args.split_seed,
        "dev_size": args.dev_size,
        "labels": LABELS,
        "counts": {
            "train": train_df["stance"].value_counts().to_dict(),
            "dev": dev_df["stance"].value_counts().to_dict(),
            "test": master_test["stance"].value_counts().to_dict(),
        },
        "n_rows": {"train": len(train_df), "dev": len(dev_df), "test": len(master_test)},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"train": train_path, "dev": dev_path, "test": test_path}


def subprocess_env(seed: int, gpus: str) -> Dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env["PYTHONHASHSEED"] = str(seed)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return env


def split_gpu_ids(gpus: str) -> List[str]:
    gpu_ids = [g.strip() for g in gpus.split(",") if g.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU id")
    return gpu_ids


def args_with_gpus(args: argparse.Namespace, gpus: str) -> argparse.Namespace:
    copied = argparse.Namespace(**vars(args))
    copied.gpus = gpus
    return copied


def run_streamed(cmd: List[str], log_path: Path, env: Dict[str, str], cwd: Path = PROJECT_ROOT) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n\n$ " + " ".join(shlex.quote(c) for c in cmd) + "\n")
        log.flush()
        print("$ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(cwd),
            env=env,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        rc = proc.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)


def run_bert(seed: int, seed_dir: Path, splits: Dict[str, Path], args: argparse.Namespace) -> Path:
    result_dir = seed_dir / "predictions" / "bert"
    pred_path = result_dir / "bert_predictions.csv"
    if args.resume and pred_path.exists():
        return pred_path
    cmd = [
        sys.executable,
        str(BERT_SCRIPT),
        "--train-csv", str(splits["train"]),
        "--dev-csv", str(splits["dev"]),
        "--test-csv", str(splits["test"]),
        "--output-dir", str(result_dir),
        "--model-dir", str(seed_dir / "models" / "bert"),
        "--log-file", str(seed_dir / "logs" / "bert.log"),
        "--prediction-file", pred_path.name,
        "--epochs", str(args.bert_epochs),
        "--batch-size", str(args.bert_batch_size),
        "--seed", str(seed),
    ]
    run_streamed(cmd, seed_dir / "logs" / "bert.console.log", subprocess_env(seed, args.gpus))
    return pred_path


def run_roberta(seed: int, seed_dir: Path, splits: Dict[str, Path], args: argparse.Namespace) -> Path:
    result_dir = seed_dir / "predictions" / "roberta"
    pred_path = result_dir / "roberta_predictions.csv"
    if args.resume and pred_path.exists():
        return pred_path
    cmd = [
        sys.executable,
        str(ROBERTA_SCRIPT),
        "--train-csv", str(splits["train"]),
        "--dev-csv", str(splits["dev"]),
        "--test-csv", str(splits["test"]),
        "--output-dir", str(result_dir),
        "--model-dir", str(seed_dir / "models" / "roberta"),
        "--prediction-file", pred_path.name,
        "--epochs", str(args.roberta_epochs),
        "--batch-size", str(args.roberta_batch_size),
        "--seed", str(seed),
    ]
    run_streamed(cmd, seed_dir / "logs" / "roberta.log", subprocess_env(seed, args.gpus))
    return pred_path


def run_pyabsa(seed: int, seed_dir: Path, splits: Dict[str, Path], args: argparse.Namespace) -> Path:
    result_dir = seed_dir / "predictions" / "pyabsa"
    pred_path = result_dir / "pyabsa_predictions.csv"
    if args.resume and pred_path.exists():
        return pred_path
    cmd = [
        sys.executable,
        str(PYABSA_SCRIPT),
        "--train-csv", str(splits["train"]),
        "--dev-csv", str(splits["dev"]),
        "--test-csv", str(splits["test"]),
        "--output-dir", str(result_dir),
        "--model-dir", str(seed_dir / "models" / "pyabsa"),
        "--dataset-dir", str(seed_dir / "pyabsa_dataset"),
        "--log-file", str(seed_dir / "logs" / "pyabsa.log"),
        "--prediction-file", pred_path.name,
        "--epochs", str(args.pyabsa_epochs),
        "--batch-size", str(args.pyabsa_batch_size),
        "--seed", str(seed),
    ]
    run_streamed(cmd, seed_dir / "logs" / "pyabsa.console.log", subprocess_env(seed, args.gpus))
    return pred_path


def _copy_deterministic_predictions(seed_dir: Path, model_name: str, pred_path: Path, file_name: str) -> bool:
    """If a sibling seed_*/predictions/<model>/<file> already has the predictions
    for a deterministic-inference model, copy it instead of re-running. Greedy
    decoding + same prompt + same shot pool ⇒ identical predictions across seeds.
    """
    output_root = seed_dir.parent
    for other in sorted(output_root.glob("seed_*")):
        if other.resolve() == seed_dir.resolve():
            continue
        candidate = other / "predictions" / model_name / file_name
        if candidate.exists():
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, pred_path)
            print(f"[info] Copied {model_name} predictions from {other.name} (deterministic across seeds)", flush=True)
            return True
    return False


def run_mistral_base(seed: int, seed_dir: Path, splits: Dict[str, Path], args: argparse.Namespace) -> Path:
    result_dir = seed_dir / "predictions" / "mistral_base"
    result_dir.mkdir(parents=True, exist_ok=True)
    pred_path = result_dir / "mistral_base_predictions.csv"
    if args.resume and pred_path.exists():
        return pred_path
    if _copy_deterministic_predictions(seed_dir, "mistral_base", pred_path, pred_path.name):
        return pred_path
    cmd = [
        sys.executable,
        str(MISTRAL_BASE_SCRIPT),
        "--model-dir", str(MISTRAL_MODEL),
        "--test-csv", str(splits["test"]),
        "--output-dir", str(result_dir),
        "--output-prefix", "mistral_base",
        "--few-shot", "0",
        "--seed", str(seed),
    ]
    run_streamed(cmd, seed_dir / "logs" / "mistral_base.log", subprocess_env(seed, args.gpus))
    return pred_path


def run_mistral_fewshot(seed: int, seed_dir: Path, splits: Dict[str, Path], args: argparse.Namespace) -> Path:
    """Few-shot inference using the base Mistral (no LoRA adapter). Uses the
    same prompter as run_mistral_lora's inference path so the only difference
    between mistral_fewshot and mistral_lora is the LoRA adapter — clean
    ablation of the fine-tune.
    """
    result_dir = seed_dir / "predictions" / "mistral_fewshot"
    result_dir.mkdir(parents=True, exist_ok=True)
    pred_path = result_dir / "mistral_fewshot_predictions.csv"
    if args.resume and pred_path.exists():
        return pred_path
    if _copy_deterministic_predictions(seed_dir, "mistral_fewshot", pred_path, pred_path.name):
        return pred_path
    cmd = [
        sys.executable,
        str(MISTRAL_INFER_SCRIPT),
        "--input_csv", str(splits["test"]),
        "--model", str(MISTRAL_MODEL),
        "--shots_dir", str(SHOTS_DIR),
        "--shots_prefix", "kyra",
        "--shots_json", str(SHOTS_JSON),
        "--output_csv", str(pred_path),
        "--batch_size", str(args.mistral_infer_batch_size),
        "--bucket_by_length",
        "--resume",
        "--log_file", str(seed_dir / "logs" / "mistral_fewshot_infer.log"),
    ]
    run_streamed(cmd, seed_dir / "logs" / "mistral_fewshot.console.log", subprocess_env(seed, args.gpus))
    return pred_path


def run_mistral_lora(seed: int, seed_dir: Path, splits: Dict[str, Path], args: argparse.Namespace) -> Path:
    result_dir = seed_dir / "predictions" / "mistral_lora"
    result_dir.mkdir(parents=True, exist_ok=True)
    pred_path = result_dir / "mistral_lora_predictions.csv"
    adapter_dir = seed_dir / "models" / "mistral_lora_adapter"

    gpu_ids = split_gpu_ids(args.gpus)
    nproc = len(gpu_ids)

    if not (args.resume and (adapter_dir / "adapter_model.safetensors").exists()):
        # Multi-GPU LoRA training via torch.distributed (DDP). Each rank loads a
        # full model copy onto its local-rank GPU; gradients are all-reduced.
        # Far faster than device_map="auto" model parallelism for a model this
        # size (Mistral-7B fits on one GPU in bf16).
        if nproc > 1:
            launcher = [
                sys.executable, "-m", "torch.distributed.run",
                "--standalone",
                "--nnodes", "1",
                "--nproc_per_node", str(nproc),
            ]
        else:
            launcher = [sys.executable]

        train_cmd = launcher + [
            str(MISTRAL_TRAIN_SCRIPT),
            "--model-dir", str(MISTRAL_MODEL),
            "--data-csv", str(splits["train"]),
            "--dev-csv", str(splits["dev"]),
            "--out-dir", str(adapter_dir),
            "--max-length", str(args.mistral_max_length),
            "--epochs", str(args.mistral_epochs),
            "--batch-size", str(args.mistral_batch_size),
            "--grad-accum", str(args.mistral_grad_accum),
            "--lr", str(args.mistral_lr),
            "--bf16",
            "--seed", str(seed),
        ]
        run_streamed(train_cmd, seed_dir / "logs" / "mistral_lora_train.log", subprocess_env(seed, args.gpus))

    if args.resume and pred_path.exists():
        return pred_path

    # Inference is single-GPU; pin to the first visible GPU.
    infer_cmd = [
        sys.executable,
        str(MISTRAL_INFER_SCRIPT),
        "--input_csv", str(splits["test"]),
        "--model", str(MISTRAL_MODEL),
        "--lora_adapter", str(adapter_dir),
        "--shots_dir", str(SHOTS_DIR),
        "--shots_prefix", "kyra",
        "--shots_json", str(SHOTS_JSON),
        "--output_csv", str(pred_path),
        "--batch_size", str(args.mistral_infer_batch_size),
        "--bucket_by_length",
        "--resume",
        "--log_file", str(seed_dir / "logs" / "mistral_lora_infer.log"),
    ]
    run_streamed(infer_cmd, seed_dir / "logs" / "mistral_lora_infer.console.log", subprocess_env(seed, gpu_ids[0]))
    return pred_path


def compute_metrics(model_name: str, pred_path: Path, seed: int) -> Dict[str, object]:
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score

    cfg = PREDICTION_CONFIG[model_name]
    df = pd.read_csv(pred_path)
    gold_col = "original_stance" if "original_stance" in df.columns else "stance"
    if gold_col not in df.columns:
        raise ValueError(f"{pred_path} does not contain original_stance or stance")
    if cfg["pred_col"] not in df.columns:
        raise ValueError(f"{pred_path} does not contain prediction column {cfg['pred_col']}")

    y_true = df[gold_col].apply(normalize_stance)
    y_pred = df[cfg["pred_col"]].apply(normalize_stance)
    valid = y_true.isin(LABELS) & y_pred.isin(LABELS)
    y_true = y_true[valid].tolist()
    y_pred = y_pred[valid].tolist()
    if not y_true:
        raise ValueError(f"No valid prediction rows found in {pred_path}")

    return {
        "seed": seed,
        "model": model_name,
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", labels=LABELS, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0),
        "prediction_file": str(pred_path),
    }


def write_summary(metrics: List[Dict[str, object]], output_root: Path) -> None:
    import pandas as pd

    metrics_df = pd.DataFrame(metrics)
    metrics_path = output_root / "metrics_by_seed.csv"
    metrics_df.to_csv(metrics_path, index=False)

    rows = []
    for model, group in metrics_df.groupby("model", sort=False):
        row = {"model": model, "seeds": ",".join(map(str, sorted(group["seed"].unique())))}
        for metric in ["accuracy", "f1_weighted", "f1_macro"]:
            mean = group[metric].mean()
            std = group[metric].std(ddof=1) if len(group) > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_mean_std"] = f"{mean:.4f} ± {std:.4f}"
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_path = output_root / "metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    md_lines = [
        "# Multi-Seed Model Comparison",
        "",
        "| Model | Accuracy | Weighted F1 | Macro F1 |",
        "|---|---:|---:|---:|",
    ]
    for _, row in summary_df.iterrows():
        md_lines.append(
            f"| {row['model']} | {row['accuracy_mean_std']} | "
            f"{row['f1_weighted_mean_std']} | {row['f1_macro_mean_std']} |"
        )
    (output_root / "metrics_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Saved per-seed metrics: {metrics_path}")
    print(f"Saved summary metrics: {summary_path}")


def validate_paths() -> None:
    required = [
        MASTER_TRAIN,
        MASTER_TEST,
        SHOTS_DIR,
        SHOTS_JSON,
        BERT_SCRIPT,
        ROBERTA_SCRIPT,
        PYABSA_SCRIPT,
        MISTRAL_TRAIN_SCRIPT,
        MISTRAL_INFER_SCRIPT,
        MISTRAL_MODEL,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))


def main() -> None:
    args = parse_args()
    if args.create_env:
        ensure_env(args.env_name)
        reexec_in_env(args)

    # Delayed imports so --create-env works even from bare system Python.
    import pandas  # noqa: F401
    import sklearn  # noqa: F401

    validate_paths()
    if len(args.seeds) < 3:
        raise ValueError("Use at least 3 seeds for the reviewer-requested stability check.")

    if args.output_root:
        output_root = Path(args.output_root).expanduser().resolve()
        if output_root.exists() and not args.resume:
            raise FileExistsError(f"{output_root} already exists. Use --resume or choose a new --output-root.")
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = MULTISEED_DIR / "runs" / f"run_{stamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["mistral_model"] = str(MISTRAL_MODEL)
    config["old_lora_adapter_not_used"] = str(PROJECT_ROOT / "lora_adaptor" / "out_train_final" / "mistral7b_stance_lora")
    (output_root / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    splits = prepare_splits(args, output_root)
    if args.prepare_only:
        print(f"Prepared fixed 3-class splits under: {output_root / 'splits'}")
        return
    runners = {
        "bert": run_bert,
        "roberta": run_roberta,
        "pyabsa": run_pyabsa,
        "mistral_base": run_mistral_base,
        "mistral_fewshot": run_mistral_fewshot,
        "mistral_lora": run_mistral_lora,
    }

    all_metrics: List[Dict[str, object]] = []
    for seed in args.seeds:
        seed_dir = output_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Seed {seed} ===", flush=True)
        if args.parallel_models:
            gpu_ids = split_gpu_ids(args.gpus)

            def _record(model_name: str, pred_path: Path) -> None:
                metric = compute_metrics(model_name, pred_path, seed)
                all_metrics.append(metric)
                model_metrics_path = seed_dir / "predictions" / model_name / "metrics.json"
                model_metrics_path.write_text(json.dumps(metric, indent=2), encoding="utf-8")
                print(
                    f"{model_name} seed={seed}: accuracy={metric['accuracy']:.4f}, "
                    f"weighted_f1={metric['f1_weighted']:.4f}, macro_f1={metric['f1_macro']:.4f}",
                    flush=True,
                )

            # Phase 1: every non-DDP model (BERT, RoBERTa, PyABSA, mistral_base,
            # mistral_fewshot) runs concurrently. With more models than GPUs,
            # static round-robin ("model i → GPU i % N") can double-book a GPU
            # if a slot frees up out of order, so dynamically pull a GPU from
            # a queue at task start and return it on task finish.
            parallel_models = [m for m in args.models if m != "mistral_lora"]
            if parallel_models:
                gpu_queue: "queue.Queue[str]" = queue.Queue()
                for g in gpu_ids:
                    gpu_queue.put(g)

                def _run_with_dynamic_gpu(model_name: str) -> Path:
                    g = gpu_queue.get()
                    print(f"--- {model_name} seed={seed} acquired GPU {g} ---", flush=True)
                    try:
                        return runners[model_name](seed, seed_dir, splits, args_with_gpus(args, g))
                    finally:
                        gpu_queue.put(g)
                        print(f"--- {model_name} seed={seed} released GPU {g} ---", flush=True)

                future_to_model = {}
                with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
                    for model_name in parallel_models:
                        print(f"--- Submitting {model_name} seed={seed} ---", flush=True)
                        future = pool.submit(_run_with_dynamic_gpu, model_name)
                        future_to_model[future] = model_name
                    for future in as_completed(future_to_model):
                        _record(future_to_model[future], future.result())

            # Phase 2: Mistral-7B LoRA gets all GPUs via DDP (torchrun launcher
            # inside run_mistral_lora). One process per GPU, full model on each,
            # gradients all-reduced — much faster than model-parallel sharding.
            if "mistral_lora" in args.models:
                print(f"\n--- Launching mistral_lora seed={seed} on GPUs {args.gpus} (DDP) ---", flush=True)
                pred_path = runners["mistral_lora"](seed, seed_dir, splits, args)
                _record("mistral_lora", pred_path)
        else:
            for model_name in args.models:
                print(f"\n--- Running {model_name} seed={seed} on GPUs {args.gpus} ---", flush=True)
                pred_path = runners[model_name](seed, seed_dir, splits, args)
                metric = compute_metrics(model_name, pred_path, seed)
                all_metrics.append(metric)
                model_metrics_path = seed_dir / "predictions" / model_name / "metrics.json"
                model_metrics_path.write_text(json.dumps(metric, indent=2), encoding="utf-8")
                print(
                    f"{model_name} seed={seed}: accuracy={metric['accuracy']:.4f}, "
                    f"weighted_f1={metric['f1_weighted']:.4f}, macro_f1={metric['f1_macro']:.4f}",
                    flush=True,
                )
        write_summary(all_metrics, output_root)

    print(f"\nDone. All outputs are under: {output_root}")


if __name__ == "__main__":
    main()
