#!/usr/bin/env python3
"""
Run temporal Mistral LoRA inference with the same prompt style used for SFT.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


INSTR_HDR = "### Instruction:\n"
INPUT_HDR = "### Input:\n"
RESP_HDR = "### Response:\n"
VALID_STANCES = {"For", "Against", "Neutral"}

THIS_FILE = Path(__file__).resolve()
TEMPORAL_DIR = THIS_FILE.parent
FINETUNING_DIR = TEMPORAL_DIR.parent
DEFAULT_SHOTS_DIR = FINETUNING_DIR / "4_a_DataProcessing" / "data_formatting" / "jsons"
DEFAULT_SHOTS_JSON = DEFAULT_SHOTS_DIR / "kyra_modi_stance.json"


def normalize_stance(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = text.replace("favour", "for").replace("favor", "for")
    text = text.replace("nuetral", "neutral").replace("oppose", "against")
    key = re.sub(r"[^a-z]+", "", text)
    if key in {"for", "positive", "support", "supports", "pro"}:
        return "For"
    if key in {"against", "negative", "anti", "opposed", "oppose", "opposes"}:
        return "Against"
    if key in {"neutral", "unrelated", "irrelevant"}:
        return "Neutral"
    return ""


def slugify_keyword(value: Any) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", str(value or "").strip().lower()).strip("_")
    return slug or "unknown"


def load_shots_file(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Few-shot JSON must contain a list of examples: {path}")

    examples: list[dict[str, str]] = []
    for i, ex in enumerate(data):
        if not isinstance(ex, dict):
            raise ValueError(f"Few-shot example #{i} in {path} is not an object.")
        entity = ex.get("entity")
        statement = ex.get("statement")
        stance = normalize_stance(ex.get("stance"))
        if entity is None or statement is None or not stance:
            raise ValueError(f"Few-shot example #{i} in {path} missing valid entity/statement/stance.")
        reason = str(ex.get("reason", "")).strip()
        if not reason:
            reason = {
                "For": "Expresses support or positive sentiment.",
                "Against": "Expresses opposition or negative sentiment.",
                "Neutral": "No clear support or opposition is expressed.",
            }[stance]
        examples.append({
            "entity": str(entity).strip(),
            "statement": str(statement).replace("\n", " ").strip(),
            "stance": stance,
            "reason": reason,
        })
    return examples


def format_fewshot_examples(examples: list[dict[str, str]]) -> str:
    blocks = []
    for n, ex in enumerate(examples, 1):
        response = json.dumps(
            {"stance": ex["stance"], "reason": ex["reason"]},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        blocks.append(
            f"Example {n}:\n"
            f"target: {ex['entity']}\n"
            f"tweet: {ex['statement']}\n"
            f"response: {response}"
        )
    return "Examples:\n" + "\n\n".join(blocks)


class FewShotProvider:
    def __init__(self, shots_dir: Path, shots_prefix: str, fallback_json: Path | None):
        self.shots_dir = shots_dir.expanduser().resolve()
        self.shots_prefix = shots_prefix
        self.fallback_json = fallback_json.expanduser().resolve() if fallback_json else None
        self._cache: dict[str, tuple[str, str]] = {}
        self._path_cache: dict[Path, str] = {}

        if not self.shots_dir.exists() or not self.shots_dir.is_dir():
            raise FileNotFoundError(f"--shots-dir not found or not a directory: {self.shots_dir}")
        if self.fallback_json is not None and not self.fallback_json.exists():
            raise FileNotFoundError(f"--shots-json fallback not found: {self.fallback_json}")

    def for_keyword(self, keyword: Any) -> tuple[str, str]:
        slug = slugify_keyword(keyword)
        if slug in self._cache:
            return self._cache[slug]

        shots_path = self.shots_dir / f"{self.shots_prefix}_{slug}_stance.json"
        if shots_path.exists():
            source_path = shots_path
        elif self.fallback_json is not None:
            source_path = self.fallback_json
        else:
            raise FileNotFoundError(
                f"No few-shot file for keyword '{keyword}' at {shots_path}, "
                "and no --shots-json fallback provided."
            )

        if source_path not in self._path_cache:
            self._path_cache[source_path] = format_fewshot_examples(load_shots_file(source_path))

        result = (self._path_cache[source_path], source_path.name)
        self._cache[slug] = result
        return result


def build_prompt(tweet: Any, keyword: Any, fewshot_examples: str = "") -> str:
    instruction = (
        "Given a tweet and a target keyword, classify the tweet's stance toward the target as exactly one of "
        '"For", "Against", or "Neutral" and provide a concise reason grounded in the tweet. '
        'Return ONLY a compact JSON object with keys "stance" and "reason" (no extra text).'
    )
    target_input = f"target: {str(keyword).strip()}\n\ntweet: {str(tweet).strip()}"
    if fewshot_examples:
        model_input = f"{fewshot_examples}\n\nNow classify the following tweet.\n\n{target_input}"
    else:
        model_input = target_input
    return f"{INSTR_HDR}{instruction}\n\n{INPUT_HDR}{model_input}\n\n{RESP_HDR}"


def extract_json(text: str) -> tuple[str, str]:
    if not text:
        return "", ""
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not match:
        return "", ""
    try:
        obj = json.loads(match.group(0))
    except Exception:
        return "", ""
    stance = normalize_stance(obj.get("stance", ""))
    reason = str(obj.get("reason", "")).strip()
    return stance, reason


def fallback_label(text: str) -> str:
    lower = (text or "").lower()
    first = re.split(r"[\s|,.;:()\[\]{}\"'\n\r\t]+", lower.strip())[0]
    stance = normalize_stance(first)
    if stance:
        return stance
    stance = normalize_stance(lower)
    if stance:
        return stance
    return "Neutral"


def load_model(model_dir: str, adapter_dir: str, bf16: bool, trust_remote_code: bool):
    dtype = torch.bfloat16 if bf16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=trust_remote_code,
    )
    model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def generate_batch(tokenizer, model, prompts: list[str], max_input_length: int, max_new_tokens: int) -> list[str]:
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    )
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    output_ids = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = output_ids[:, enc["input_ids"].shape[1] :]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--lora-adapter", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--shots-dir", default=str(DEFAULT_SHOTS_DIR),
                        help="Directory with per-keyword few-shot JSON files like <prefix>_<keyword>_stance.json.")
    parser.add_argument("--shots-prefix", default="kyra",
                        help="Prefix for per-keyword few-shot JSON files.")
    parser.add_argument("--shots-json", default=str(DEFAULT_SHOTS_JSON),
                        help="Fallback few-shot JSON when a per-keyword file is missing.")
    parser.add_argument("--no-fewshot", action="store_true",
                        help="Disable few-shot examples and use the original bare temporal prompt.")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input_csv).expanduser().resolve()
    output_path = Path(args.output_csv).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume and output_path.exists():
        df = pd.read_csv(output_path, low_memory=False)
        print(f"[info] Resuming from {output_path}", flush=True)
        for col in ["lora_raw", "lora_stance", "lora_reason", "lora_shots_json"]:
            if col not in df.columns:
                df[col] = ""
    else:
        df = pd.read_csv(input_path, low_memory=False)
        for col in ["lora_raw", "lora_stance", "lora_reason", "lora_shots_json"]:
            df[col] = ""

    if not {"tweet", "keyword"}.issubset(df.columns):
        raise ValueError("Input CSV must contain tweet and keyword columns.")

    done = df["lora_stance"].map(normalize_stance).isin(VALID_STANCES)
    pending = df.index[~done].tolist()
    print(f"[info] Rows total={len(df)} pending={len(pending)}", flush=True)
    if not pending:
        df.to_csv(output_path, index=False)
        return

    shot_provider = None
    if not args.no_fewshot:
        shot_provider = FewShotProvider(
            Path(args.shots_dir),
            args.shots_prefix,
            Path(args.shots_json) if args.shots_json else None,
        )
        print(f"[info] Few-shot directory={shot_provider.shots_dir}", flush=True)
        print(f"[info] Few-shot fallback={shot_provider.fallback_json}", flush=True)
    else:
        print("[info] Few-shot examples disabled.", flush=True)

    print(f"[info] Loading model={args.model_dir}", flush=True)
    print(f"[info] Loading adapter={args.lora_adapter}", flush=True)
    tokenizer, model = load_model(args.model_dir, args.lora_adapter, args.bf16, args.trust_remote_code)

    started = time.time()
    batch_size = max(1, int(args.batch_size))
    for start in tqdm(range(0, len(pending), batch_size), desc="Temporal LoRA inference"):
        batch_idx = pending[start : start + batch_size]
        prompts: list[str] = []
        shot_sources: list[str] = []
        for i in batch_idx:
            if shot_provider is not None:
                fewshot_examples, shots_src = shot_provider.for_keyword(df.at[i, "keyword"])
            else:
                fewshot_examples, shots_src = "", ""
            prompts.append(build_prompt(df.at[i, "tweet"], df.at[i, "keyword"], fewshot_examples))
            shot_sources.append(shots_src)
        texts = generate_batch(tokenizer, model, prompts, args.max_input_length, args.max_new_tokens)

        for i, text, shots_src in zip(batch_idx, texts, shot_sources):
            stance, reason = extract_json(text)
            if not stance:
                stance = fallback_label(text)
            df.at[i, "lora_raw"] = text
            df.at[i, "lora_stance"] = stance
            df.at[i, "lora_reason"] = reason
            df.at[i, "lora_shots_json"] = shots_src

        if start and start % (batch_size * 20) == 0:
            df.to_csv(output_path, index=False)

    df.to_csv(output_path, index=False)
    elapsed = time.time() - started
    print(f"[info] Wrote {output_path} in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
