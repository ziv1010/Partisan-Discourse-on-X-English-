#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Few-shot stance classification (entity/statement) — FINAL JSON OUTPUT — BATCHED + RESUME
+ LoRA support (Peft) to run base or finetuned.
+ Per-keyword few-shot selection from a directory, with optional global fallback.

Task: 3-way stance only.
The model must classify each tweet toward a target keyword as exactly one of:
    - "favor"
    - "against"
    - "neutral"
There is NO "unrelated" class anymore.

- Few-shot only.
- Batched generation via raw HF pipeline(list_of_prompts, batch_size=...).
- Resume: --resume continues from existing --output_csv, skipping rows with a filled fewshot_label.
- Model returns ONLY JSON: {"stance":"<label>","reason":"<short phrase>"}.
- Prevents prompt-echo (return_full_text=False); deterministic decoding.
- Robust JSON extraction + hardened fallback normalizer.
- Adds mapped column (still stored in fewshot_label_for_against, now just repeats favor/against/neutral).
- NEW: --lora_adapter (optional). If set, loads base model from --model and applies LoRA adapter.
       Use --merge_lora to merge & unload for inference (optional).
- NEW: --shots_dir + --shots_prefix for per-keyword few-shot files named <prefix>_<keyword>_stance.json.
       Falls back to --shots_json (single file) if the per-keyword file is missing.
- NEW: writes the source JSON filename used per row in column `fewshot_shots_json`.

Usage:
python finetune.py \
  --input_csv /path/to/your.csv \
  --model /path/to/base_model_or_snapshot \
  --shots_dir /path/to/shots_dir \
  --shots_prefix kyra \
  --shots_json /path/to/fallback_few_shot_examples.json \
  --output_csv ./results/out.csv \
  --max_new_tokens 48 \
  --batch_size 16 \
  --bucket_by_length \
  --resume \
  --save_every 100 \
  [--lora_adapter /path/to/lora_adapter_dir] [--merge_lora] [--trust_remote_code]
"""

import os, json, argparse, time, logging, warnings, re
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    pipeline,
)

# LangChain imports (new style first, fallback to older) — used only for prompt construction
try:
    from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
except Exception:
    from langchain.prompts import PromptTemplate, FewShotPromptTemplate

warnings.filterwarnings(
    "ignore",
    message="You seem to be using the pipelines sequentially on GPU."
)

# ---------------------------
# Helpers: labels & keyword slug
# ---------------------------

def _canon_label(s: str) -> str:
    return re.sub(r"[^a-z]+", "", (s or "").strip().lower())

# Accepts older few-shot JSONs that may use supports/denies/for/against/etc.
# We collapse everything into {favor, against, neutral}.
_LABEL_MAP_IN = {
    "positive": "favor",
    "for": "favor",
    "support": "favor",
    "supports": "favor",
    "pro": "favor",
    "favor": "favor",
    "favour": "favor",
    "favors": "favor",
    "favourable": "favor",
    "favorable": "favor",

    "negative": "against",
    "denies": "against",
    "deny": "against",
    "against": "against",
    "anti": "against",
    "oppose": "against",
    "opposes": "against",
    "opposed": "against",
    "con": "against",

    "neutral": "neutral",
    "unrelated": "neutral",   # legacy data -> treat as neutral
    "irrelevant": "neutral",  # legacy data -> treat as neutral
}
_VALID = {"favor", "against", "neutral"}

def _normalize_shot_label(s: str) -> str:
    c = _canon_label(s)
    if c in _VALID:
        return c
    if c in _LABEL_MAP_IN:
        return _LABEL_MAP_IN[c]
    return "neutral"

def _slugify_kw(s: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", (s or "").strip().lower()).strip("_")
    return slug or "unknown"

# ---------------------------
# HF model wrapper (RAW PIPELINE for batching) + LoRA support
# ---------------------------

def build_hf_pipe(
    model_path: str,
    task_hint: str = "auto",
    max_new_tokens: int = 48,
    lora_adapter: Optional[str] = None,
    merge_lora: bool = False,
    trust_remote_code: bool = False,
):
    """
    Returns a raw HF pipeline that supports batched calls: pipe(list_of_prompts, batch_size=...)

    Tokenizer controls padding/truncation; we do NOT pass padding/truncation into the pipeline call,
    to avoid leaking kwargs into model.generate() on older Transformers.
    """
    # (Optional) faster matmul on Ampere+
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # Resolve config to decide decoder vs encoder-decoder without instantiating twice
    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        is_seq2seq = bool(getattr(cfg, "is_encoder_decoder", False))
    except Exception:
        # Fallback to heuristic
        is_seq2seq = False

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, legacy=True, trust_remote_code=trust_remote_code
    )

    # ensure pad token & left-padding for decoder-only models
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if not is_seq2seq:
        tokenizer.padding_side = "left"
    # truncate long prompts from the left to fit context windows of typical LLMs
    tokenizer.truncation_side = "left"
    # tame pathological model_max_length values (some tokenizers set it to 1e30)
    try:
        if getattr(tokenizer, "model_max_length", None) and tokenizer.model_max_length > 8192:
            tokenizer.model_max_length = 8192
    except Exception:
        pass

    # Task selection
    if task_hint in ("text-generation", "text2text-generation"):
        task = task_hint
    else:
        task = "text2text-generation" if is_seq2seq else "text-generation"

    # Prefer bf16 on supported GPUs; else fp16; else float32 (CPU)
    use_cuda = torch.cuda.is_available()
    torch_dtype = None
    if use_cuda:
        try:
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float16
        except Exception:
            torch_dtype = torch.float16

    # Load base model
    common_kwargs = dict(
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    
    # Try to enable Flash Attention 2 for faster inference (requires flash-attn package)
    try:
        import flash_attn
        common_kwargs["attn_implementation"] = "flash_attention_2"
        logging.getLogger("fewshot_stance_json").info("Flash Attention 2 enabled for faster inference")
    except ImportError:
        pass  # flash-attn not installed, use default attention
    
    if is_seq2seq:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **common_kwargs)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(model_path, **common_kwargs)

    # Optionally attach LoRA adapter
    model = base_model
    if lora_adapter:
        try:
            from peft import PeftModel
        except Exception as e:
            raise RuntimeError(
                "peft is required to use --lora_adapter. Install with: pip install peft"
            ) from e

        model = PeftModel.from_pretrained(base_model, lora_adapter, is_trainable=False)
        if merge_lora:
            # Merge LoRA weights into base for slightly faster inference
            model = model.merge_and_unload()

    # Build pipeline kwargs
    kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        # Deterministic decoding
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )
    if task == "text-generation":
        kwargs["return_full_text"] = False

    pipe = pipeline(task, **kwargs)

    # Ensure generation uses pad_token_id we set (harmless no-op if already set)
    try:
        pipe.model.generation_config.pad_token_id = pipe.tokenizer.pad_token_id
        pipe.model.generation_config.eos_token_id = pipe.tokenizer.eos_token_id
    except Exception:
        pass

    return pipe

# ---------------------------
# Few-shot examples + label utils
# ---------------------------

def load_shots_file(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for i, ex in enumerate(data):
        entity = ex.get("entity")
        statement = ex.get("statement")
        stance = ex.get("stance")
        if entity is None or statement is None or stance is None:
            raise ValueError(f"Few-shot example #{i} in {path} missing required keys.")
        stance = _normalize_shot_label(stance)
        if stance not in _VALID:
            stance = "neutral"
        out.append({"entity": entity, "statement": statement, "stance": stance})
    return out

def normalize_label(text: str) -> str:
    """
    Try to coerce arbitrary model output into one of:
      favor / against / neutral
    """
    t = (text or "").strip().lower()

    # first token heuristic
    first = re.split(r'[\s\|\.,:;()\[\]\{\}\n\r\t"]+', t)[0]
    if first in {"favor", "favour", "against", "neutral"}:
        if first == "favour":
            return "favor"
        return first

    # regex-based fallbacks
    if re.search(r'\b(favor|favour|support|supports|pro|in\s+favor|for)\b', t):
        return "favor"
    if re.search(r'\b(against|anti|oppose|opposes|opposed|deny|denies|con)\b', t):
        return "against"
    if re.search(r'\bneutral\b', t):
        return "neutral"

    # default
    return "neutral"

def extract_json(s: str):
    """
    Extract {"stance": "...", "reason": "..."} from model output
    and validate stance in {favor, against, neutral}.
    """
    if not s:
        return None, None
    try:
        # strip ```json ... ``` if present
        mcode = re.search(r"```json(.*?)```", s, flags=re.DOTALL | re.IGNORECASE)
        if mcode:
            s = mcode.group(1)
        m = re.search(r'\{.*?\}', s, flags=re.DOTALL)
        if not m:
            return None, None
        obj = json.loads(m.group(0))
        stance = obj.get("stance")
        reason = obj.get("reason", "")
        if stance in {"favor", "favour", "against", "neutral"}:
            if stance == "favour":
                stance = "favor"
            return stance, str(reason)
    except Exception:
        pass
    return None, None

# ---------------------------
# Prompt construction
# ---------------------------

def make_few_shot_prompt(examples):
    example_template = (
        "entity: {entity}\n"
        "statement: {statement}\n"
        "stance: {stance}"
    )
    example_prompt = PromptTemplate(
        input_variables=["entity", "statement", "stance"],
        template=example_template,
    )

    prefix = (
        "Stance classification is the task of determining the expressed or implied opinion, "
        "or stance, of a statement toward a certain, specified target. The following "
        "statements are social media posts expressing opinions about entities. "
        "Each statement can either be in favor of the entity, against the entity, or neutral."
    )

    # NOTE: braces in the example JSON are escaped as {{ and }} to keep them literal.
    suffix = (
        "Analyze the following social media statement and determine its stance towards the provided entity.\n"
        "Return ONLY a compact JSON object with exactly these keys:\n"
        '- \"stance\": one of \"favor\", \"against\", \"neutral\"\n'
        '- \"reason\": a short phrase (not a paragraph)\n'
        'Example: {{\"stance\":\"favor\",\"reason\":\"praises the policy\"}}\n'
        "entity: {event}\n"
        "statement: {statement}\n"
        "JSON:"
    )

    return FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["event", "statement"],
        example_separator="\n\n",
    )

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Few-shot stance classifier (entity/statement) — JSON output (batched + resume) with optional LoRA and per-keyword shots. 3-way labels: favor/against/neutral.")
    ap.add_argument("--input_csv", required=True, help="Input CSV with columns: tweet,...,keyword")
    ap.add_argument("--model", required=True, help="Base model dir/checkpoint (decoder or encoder-decoder).")

    # Per-keyword few-shot config
    ap.add_argument("--shots_dir", required=True, help="Directory containing per-keyword few-shot JSONs like <prefix>_<keyword>_stance.json")
    ap.add_argument("--shots_prefix", default="kyra", help="Prefix for few-shot JSON filenames (default: kyra)")
    # Fallback (optional) single shots file if per-keyword file is missing
    ap.add_argument("--shots_json", default=None, help="Fallback JSON with few-shot examples when per-keyword file is missing.")

    ap.add_argument("--output_csv", required=True, help="Output CSV path.")
    ap.add_argument("--task_hint", default="auto", choices=["auto", "text-generation", "text2text-generation"],
                    help="Force HF pipeline task if needed.")
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--batch_size", type=int, default=64, help="Batch size for inference. Increase for multi-GPU setups.")
    ap.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU IDs to use, e.g., '0,1,2,3'. If not set, uses all available.")
    ap.add_argument("--bucket_by_length", action="store_true", help="Sort prompts by length within the run for less padding.")
    ap.add_argument("--resume", action="store_true", help="Resume from existing --output_csv if present; skip already-scored rows.")
    ap.add_argument("--save_every", type=int, default=100, help="Write partial CSV every N processed rows.")
    ap.add_argument("--log_file", default=None)

    # LoRA options
    ap.add_argument("--lora_adapter", default=None, help="Path to a LoRA adapter directory (PEFT). If set, runs finetuned model.")
    ap.add_argument("--merge_lora", action="store_true", help="Merge LoRA weights into base for inference (optional).")
    ap.add_argument("--trust_remote_code", action="store_true", help="Pass trust_remote_code=True to HF loaders (needed for some models).")

    args = ap.parse_args()

    # Logging
    logger = logging.getLogger("fewshot_stance_json")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    if args.log_file:
        fh = logging.FileHandler(args.log_file)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

    mode_str = "FINETUNED (LoRA)" if args.lora_adapter else "BASE (normal)"
    logger.info(f"Running mode: {mode_str}")

    # GPU selection - set CUDA_VISIBLE_DEVICES before any CUDA operations
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        logger.info(f"Using GPUs: {args.gpus}")
    elif torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"Using all {num_gpus} available GPU(s)")
    
    # Log GPU memory info
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({mem_total:.1f} GB)")

    shots_dir = Path(args.shots_dir).expanduser().resolve()
    if not shots_dir.exists() or not shots_dir.is_dir():
        raise FileNotFoundError(f"--shots_dir not found or not a directory: {shots_dir}")

    # Optional global fallback shots
    fallback_shots = None
    fallback_basename = None
    if args.shots_json:
        fb_path = Path(args.shots_json).expanduser().resolve()
        if fb_path.exists():
            try:
                fallback_shots = load_shots_file(fb_path)
                fallback_basename = fb_path.name
                logger.info(f"Loaded fallback shots: {fb_path} (n={len(fallback_shots)})")
            except Exception as e:
                logger.warning(f"Failed to load fallback shots {fb_path}: {e}")
        else:
            logger.warning(f"Fallback --shots_json not found: {fb_path}")

    # Load data
    df = pd.read_csv(args.input_csv)
    # Normalize columns & fill NaNs for robust processing
    if not {"tweet", "keyword"}.issubset(df.columns):
        raise ValueError("Input CSV must contain at least 'tweet' and 'keyword' columns.")
    df[["tweet", "keyword"]] = df[["tweet", "keyword"]].fillna("")

    # Output columns
    raw_col        = "fewshot_raw"
    norm_col       = "fewshot_label"               # favor | against | neutral
    mapped_col     = "fewshot_label_for_against"   # kept for backwards compat; now just favor/against/neutral
    reason_col     = "fewshot_reason"              # short explanation (from JSON)
    shots_src_col  = "fewshot_shots_json"          # basename of the JSON used for this row

    # Prepare outputs (with resume support)
    if args.resume and os.path.exists(args.output_csv):
        try:
            df_out = pd.read_csv(args.output_csv)
            if len(df_out) != len(df):
                logger.warning("Resume requested but output length != input length. Starting fresh.")
                df_out = df.copy()
            else:
                # Ensure columns exist
                if "stance_gold" not in df_out.columns:
                    df_out["stance_gold"] = np.nan
                for c in (raw_col, norm_col, mapped_col, reason_col, shots_src_col):
                    if c not in df_out.columns:
                        df_out[c] = ""
                logger.info(f"Resuming from existing file: {args.output_csv}")
        except Exception as e:
            logger.warning(f"Failed to read existing output for resume ({e}). Starting fresh.")
            df_out = df.copy()
    else:
        df_out = df.copy()

    # If fresh, initialize columns
    if "stance_gold" not in df_out.columns:
        df_out["stance_gold"] = np.nan
    for c in (raw_col, norm_col, mapped_col, reason_col, shots_src_col):
        if c not in df_out.columns:
            df_out[c] = ""

    # Build pipeline (batched) — now with optional LoRA
    pipe = build_hf_pipe(
        args.model,
        task_hint=args.task_hint,
        max_new_tokens=args.max_new_tokens,
        lora_adapter=args.lora_adapter,
        merge_lora=args.merge_lora,
        trust_remote_code=args.trust_remote_code,
    )

    # ---- Per-keyword shots/template + source cache (lazy) ----
    tmpl_cache: Dict[str, FewShotPromptTemplate] = {}
    src_cache: Dict[str, str] = {}  # slug -> basename of JSON used

    def get_prompt_for_keyword(keyword_value: str) -> Tuple[FewShotPromptTemplate, str]:
        slug = _slugify_kw(keyword_value)
        if slug in tmpl_cache:
            return tmpl_cache[slug], src_cache[slug]
        # Resolve per-keyword file
        shots_path = shots_dir / f"{args.shots_prefix}_{slug}_stance.json"
        if shots_path.exists():
            shots = load_shots_file(shots_path)
            tmpl_cache[slug] = make_few_shot_prompt(shots)
            src_cache[slug] = shots_path.name
            return tmpl_cache[slug], src_cache[slug]
        # Fallback to global shots if available
        if fallback_shots is not None:
            tmpl_cache[slug] = make_few_shot_prompt(fallback_shots)
            src_cache[slug] = fallback_basename or "fallback_unknown.json"
            return tmpl_cache[slug], src_cache[slug]
        raise FileNotFoundError(
            f"No few-shot file for keyword '{keyword_value}' (slug '{slug}') at {shots_path}, "
            f"and no --shots_json fallback provided."
        )

    # Determine which rows still need scoring
    valid_labels = {"favor", "against", "neutral"}
    if "fewshot_label" in df_out.columns:
        done_mask = df_out["fewshot_label"].astype(str).str.strip().str.lower().isin(valid_labels)
    else:
        done_mask = pd.Series(False, index=df_out.index)

    remaining_idx = df_out.index[~done_mask].tolist()
    if len(remaining_idx) == 0:
        logger.info("All rows already scored. Nothing to do.")
        df_out.to_csv(args.output_csv, index=False)
        return
    # Lazy per-keyword loading: we load the shots file only when a row with that keyword is processed.

    # Build prompts only for remaining rows (lazy, per-keyword)
    prompts: List[str] = []
    row_indices: List[int] = []
    row_sources: List[str] = []  # basename per row

    for i in remaining_idx:
        row = df_out.loc[i]
        entity = str(row["keyword"]).strip()
        statement = str(row["tweet"]).replace("\n", " ").strip()
        try:
            prompt_tmpl, src_basename = get_prompt_for_keyword(entity)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Missing few-shot file for keyword '{entity}' and no --shots_json fallback."
            ) from e
        prompts.append(prompt_tmpl.format(event=entity, statement=statement))
        row_indices.append(i)
        row_sources.append(src_basename)

    # Optional: bucket by length to reduce padding (keep sources aligned)
    if args.bucket_by_length:
        order = sorted(range(len(prompts)), key=lambda k: len(prompts[k]))
        prompts     = [prompts[k] for k in order]
        row_indices = [row_indices[k] for k in order]
        row_sources = [row_sources[k] for k in order]

    # Batched inference
    start = time.time()
    processed = 0
    last_saved = 0
    B = max(1, int(args.batch_size))

    try:
        for start_idx in tqdm(range(0, len(prompts), B), desc="Scoring (batched)"):
            batch_prompts = prompts[start_idx:start_idx + B]
            batch_rows    = row_indices[start_idx:start_idx + B]
            batch_srcs    = row_sources[start_idx:start_idx + B]

            # NOTE: Do NOT pass padding/truncation here; they leak into model.generate on older versions.
            outs = pipe(batch_prompts, batch_size=B)

            # Normalize outputs to strings
            texts: List[str] = []
            for out in outs:
                # Depending on pipeline version, out can be a list[{"generated_text": ...}] or dict
                if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
                    text = out[0].get("generated_text") or out[0].get("summary_text") or str(out[0])
                elif isinstance(out, dict):
                    text = out.get("generated_text") or out.get("summary_text") or str(out)
                else:
                    text = str(out)
                texts.append(text)

            # Parse + write back
            label_map = {
                "favor": "favor",
                "against": "against",
                "neutral": "neutral",
            }

            for i_row, text, src_name in zip(batch_rows, texts, batch_srcs):
                stance_json, reason = extract_json(text)
                if stance_json:
                    norm = stance_json
                    df_out.at[i_row, reason_col] = (reason or "").strip()
                else:
                    norm = normalize_label(text)

                df_out.at[i_row, raw_col]        = text
                df_out.at[i_row, norm_col]       = norm
                df_out.at[i_row, mapped_col]     = label_map.get(norm, "neutral")
                df_out.at[i_row, shots_src_col]  = src_name  # record which JSON was used

            processed += len(batch_rows)
            if processed - last_saved >= args.save_every:
                df_out.to_csv(args.output_csv, index=False)
                last_saved = processed
    finally:
        # Ensure we always persist partial progress on interrupts/exceptions
        df_out.to_csv(args.output_csv, index=False)

    # Final save (redundant but explicit)
    df_out.to_csv(args.output_csv, index=False)
    elapsed = time.time() - start
    logger.info(f"Done. Scored {processed} rows this run in {elapsed:.1f}s. Wrote: {args.output_csv}")

if __name__ == "__main__":
    main()
