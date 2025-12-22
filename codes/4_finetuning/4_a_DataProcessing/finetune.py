# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# QLoRA fine-tuning for Mistral-7B-Instruct (offline), on stance+reason tweets CSV.

# - Model path (offline): /home/ziv.barretto_ug25/LLM/mistral_7b_instruct_v0_3_full
# - Expects CSV columns (we normalize headers): tweet, keyword, stance, stance_reason
#   (Your sample's "stance reason" will be auto-normalized to "stance_reason".)
# - Trains to emit strict JSON: {"stance": "<For|Against|Neutral|Unrelated>", "reason": "<short reason>"}
# - Masks loss for the prompt, optimizes only the JSON completion.
# - Uses TRL SFTTrainer if available; otherwise falls back to HF Trainer with manual masking.

# Run (example):
#   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \\
#   python finetune_mistral_stance.py \\
#     --model-dir /home/ziv.barretto_ug25/LLM/mistral_7b_instruct_v0_3_full \\
#     --data-csv /path/to/stance_tweets.csv \\
#     --out-dir /home/ziv.barretto_ug25/LLM/out/mistral7b_stance_lora

# Notes:
# - Keep the same prompt format for inference.
# - For inference on 4-bit, load base 4-bit + adapter (don’t merge); if you must merge, merge into fp16/bf16 base.
# """

# import os, argparse, math, json, re
# from typing import Dict, List, Tuple

# os.environ.setdefault("HF_HUB_OFFLINE", "1")
# os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# import torch
# from datasets import load_dataset
# from transformers import (
#     AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, default_data_collator,
#     BitsAndBytesConfig,
# )
# from peft import LoraConfig, get_peft_model, TaskType, PeftModel, prepare_model_for_kbit_training

# # Optional TRL path (preferred)
# try:
#     from trl import SFTTrainer
#     from trl import DataCollatorForCompletionOnlyLM  # present in recent TRL
#     HAS_TRL = True
# except Exception:
#     HAS_TRL = False


# # -------------------------- Prompting --------------------------

# INSTR_HDR = "### Instruction:\n"
# INPUT_HDR = "### Input:\n"
# RESP_HDR  = "### Response:\n"

# ALLOWED_STANCES = {"for": "For", "against": "Against", "neutral": "Neutral", "unrelated": "Unrelated"}

# def normalize_header(name: str) -> str:
#     # strip spaces/commas, lower, replace spaces with underscore
#     n = name.strip().strip(",").lower()
#     n = re.sub(r"\s+", "_", n)
#     return n

# def normalize_stance(x: str) -> str:
#     if x is None:
#         return "Unrelated"
#     s = str(x).strip().lower()
#     s = s.replace(".", "")
#     s = s.replace("favour", "for")
#     s = s.replace("in_favor", "for")
#     s = s.replace("in favour", "for")
#     s = s.replace("oppose", "against")
#     s = s.strip()
#     return ALLOWED_STANCES.get(s, "Unrelated")  # default if weird label

# def build_prompt_and_response(row: Dict) -> Tuple[str, str]:
#     tweet   = str(row.get("tweet", "")).strip()
#     keyword = str(row.get("keyword", "")).strip()
#     stance  = normalize_stance(row.get("stance", ""))
#     reason  = str(row.get("stance_reason", "")).strip()

#     if not reason:
#         # Backstop if dataset reason is empty
#         if stance == "Neutral":
#             reason = "No clear support or opposition is expressed."
#         elif stance == "For":
#             reason = "Expresses support or positive sentiment."
#         elif stance == "Against":
#             reason = "Expresses opposition or negative sentiment."
#         else:
#             reason = "No stance toward the target is expressed."

#     # Instruction keeps it deterministic with strict JSON output
#     instruction = (
#         "Given a tweet and a target keyword, classify the tweet’s stance toward the target as one of "
#         "\"For\", \"Against\", \"Neutral\", or \"Unrelated\" and provide a concise reason grounded in the tweet. "
#         "Return ONLY a compact JSON object with keys \"stance\" and \"reason\" (no extra text)."
#     )

#     # Input includes target to condition stance on
#     model_input = f"target: {keyword}\n\ntweet: {tweet}"

#     gold_json = json.dumps({"stance": stance, "reason": reason}, ensure_ascii=False)

#     prompt   = f"{INSTR_HDR}{instruction}\n\n{INPUT_HDR}{model_input}\n\n{RESP_HDR}"
#     target   = gold_json  # what we want the model to generate
#     return prompt, target


# # -------------------------- Dataset prep --------------------------

# def load_and_prepare_csv(csv_path: str):
#     # datasets will infer schema; we’ll normalize column names
#     ds = load_dataset("csv", data_files=csv_path)["train"]

#     # normalize headers: rename columns in-place
#     cols = ds.column_names
#     mapping = {c: normalize_header(c) for c in cols}
#     ds = ds.rename_columns(mapping)

#     # Minimal column sanity
#     for req in ("tweet", "keyword", "stance"):
#         if req not in ds.column_names:
#             raise ValueError(f"Required column '{req}' not found after normalization. Got: {ds.column_names}")

#     # optional column 'stance_reason' (from 'stance reason' in your CSV)
#     if "stance_reason" not in ds.column_names:
#         ds = ds.add_column("stance_reason", [""] * len(ds))

#     # format to a single text field the SFTTrainer can consume
#     def to_text_examples(batch):
#         texts = []
#         for t, k, s, r in zip(batch["tweet"], batch["keyword"], batch["stance"], batch["stance_reason"]):
#             prompt, target = build_prompt_and_response(
#                 {"tweet": t, "keyword": k, "stance": s, "stance_reason": r}
#             )
#             texts.append(prompt + target)
#         return {"text": texts}

#     ds = ds.map(to_text_examples, batched=True, remove_columns=ds.column_names)
#     return ds


# # -------------------------- Tokenization & Masking --------------------------

# def make_tokenizer(model_dir: str):
#     tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
#     if tok.pad_token is None:
#         tok.pad_token = tok.eos_token
#     tok.padding_side = "right"
#     return tok

# def make_data_collator_for_response_only(tokenizer):
#     """
#     Prefer TRL’s DataCollatorForCompletionOnlyLM which automatically masks everything
#     before RESP_HDR. If TRL is unavailable, return None and we’ll use manual masking.
#     """
#     if not HAS_TRL:
#         return None
#     return DataCollatorForCompletionOnlyLM(
#         response_template=RESP_HDR,
#         tokenizer=tokenizer,
#     )

# def manual_tokenize_with_mask(dataset, tokenizer, max_length: int):
#     """
#     Fallback when TRL is unavailable. We tokenize and set labels=-100 up to and including RESP_HDR.
#     """
#     def tokenize(batch):
#         input_texts = batch["text"]
#         enc = tokenizer(
#             input_texts,
#             padding="max_length",
#             truncation=True,
#             max_length=max_length,
#             return_tensors="pt",
#             add_special_tokens=True,
#         )

#         labels = enc["input_ids"].clone()
#         # find RESP_HDR position per sample using tokenizer’s offsets
#         # simple heuristic: re-tokenize the prompt portion length
#         prompts = []
#         for text in input_texts:
#             # split exactly once on RESP_HDR
#             if RESP_HDR in text:
#                 prompt = text.split(RESP_HDR, 1)[0] + RESP_HDR
#             else:
#                 prompt = text
#             prompts.append(prompt)

#         prompt_ids = tokenizer(
#             prompts,
#             padding=False,
#             truncation=True,
#             max_length=max_length,
#             add_special_tokens=True,
#         )["input_ids"]

#         for i, p_ids in enumerate(prompt_ids):
#             pl = len(p_ids)
#             labels[i, :pl] = -100

#         return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}

#     tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
#     tokenized.set_format(type="torch")
#     return tokenized


# # -------------------------- Model & LoRA --------------------------

# def load_qlora_model(model_dir: str):
#     bnb = BitsAndBytesConfig(load_in_8bit=True)  # << use 8-bit
#     model = AutoModelForCausalLM.from_pretrained(
#         model_dir,
#         device_map="auto",
#         quantization_config=bnb,
#         trust_remote_code=True,
#     )
#     model.gradient_checkpointing_enable()
#     model.config.use_cache = False
#     from peft import prepare_model_for_kbit_training
#     model = prepare_model_for_kbit_training(model)
#     return model


# # -------------------------- PPL (response-only) --------------------------

# @torch.no_grad()
# def eval_perplexity_response_only(model, tokenizer, raw_ds, max_length=1024, batch_size=4):
#     # Retokenize with labels masked to response
#     tmp = manual_tokenize_with_mask(raw_ds, tokenizer, max_length)
#     loader = torch.utils.data.DataLoader(tmp, batch_size=batch_size, collate_fn=default_data_collator)
#     losses = []
#     device = next(model.parameters()).device
#     model.eval()
#     for batch in loader:
#         batch = {k: v.to(device) for k, v in batch.items()}
#         loss = model(**batch).loss
#         losses.append(loss.item())
#     return math.exp(sum(losses) / max(1, len(losses)))


# # -------------------------- Main --------------------------

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--model-dir", required=True, help="Path to offline Mistral-7B-Instruct v0.3 dir")
#     ap.add_argument("--data-csv", required=True, help="Path to stance CSV")
#     ap.add_argument("--out-dir", required=True, help="Where to save LoRA adapter")
#     ap.add_argument("--max-length", type=int, default=1024)
#     ap.add_argument("--epochs", type=int, default=3)
#     ap.add_argument("--batch-size", type=int, default=2)
#     ap.add_argument("--grad-accum", type=int, default=8)
#     ap.add_argument("--lr", type=float, default=2e-4)
#     ap.add_argument("--bf16", action="store_true", help="Use bfloat16 compute (recommended on Ampere+)")
#     ap.add_argument("--eval-ppl", action="store_true", help="Compute response-only PPL after training")
#     args = ap.parse_args()

#     os.makedirs(args.out_dir, exist_ok=True)

#     print(f"[info] Loading CSV: {args.data_csv}")
#     ds = load_and_prepare_csv(args.data_csv)

#     # Split train/val (90/10) deterministically
#     ds = ds.train_test_split(test_size=0.1, seed=42)
#     train_ds, val_ds = ds["train"], ds["test"]

#     print(f"[info] Loading tokenizer & model from: {args.model_dir}")
#     tokenizer = make_tokenizer(args.model_dir)
#     model = load_qlora_model(args.model_dir)

#     total_train_tokens = None  # optional, for logging
#     use_trl = HAS_TRL

#     # -------- Training --------
#     training_args = TrainingArguments(
#         output_dir=args.out_dir,
#         per_device_train_batch_size=args.batch_size,
#         gradient_accumulation_steps=args.grad_accum,
#         num_train_epochs=args.epochs,
#         learning_rate=args.lr,
#         logging_steps=25,
#         save_strategy="epoch",
#         evaluation_strategy="epoch",
#         bf16=args.bf16,
#         fp16=(not args.bf16),
#         report_to="none",
#         remove_unused_columns=False,
#     )

#     if use_trl:
#         print("[info] Using TRL SFTTrainer with response-only masking")
#         collator = make_data_collator_for_response_only(tokenizer)
#         trainer = SFTTrainer(
#             model=model,
#             tokenizer=tokenizer,
#             args=training_args,
#             train_dataset=train_ds,
#             eval_dataset=val_ds,
#             max_seq_length=args.max_length,
#             packing=False,  # keep single example per sequence for clarity
#             formatting_func=lambda rec: rec["text"],  # dataset already has "text"
#             data_collator=collator,
#         )
#     else:
#         print("[info] TRL not found. Falling back to manual masking + HF Trainer")
#         train_tok = manual_tokenize_with_mask(train_ds, tokenizer, args.max_length)
#         val_tok   = manual_tokenize_with_mask(val_ds, tokenizer, args.max_length)
#         trainer = Trainer(
#             model=model,
#             args=training_args,
#             train_dataset=train_tok,
#             eval_dataset=val_tok,
#             tokenizer=tokenizer,
#             data_collator=default_data_collator,
#         )

#     trainer.train()
#     print("[info] Saving LoRA adapter to:", args.out_dir)
#     trainer.model.save_pretrained(args.out_dir)
#     tokenizer.save_pretrained(args.out_dir)

#     if args.eval_ppl:
#         print("[info] Computing response-only perplexity on validation split …")
#         ppl = eval_perplexity_response_only(trainer.model, tokenizer, val_ds, args.max_length, batch_size=4)
#         print(f"[metric] Val PPL (response-only): {ppl:.3f}")

#     # Small sanity-generation on 3 samples
#     try:
#         from peft import PeftModel
#         # Use base 4-bit + adapter for inference (no merge)
#         base = AutoModelForCausalLM.from_pretrained(
#             args.model_dir,
#             device_map="auto",
#             quantization_config=BitsAndBytesConfig(load_in_8bit=True),
#             trust_remote_code=True,
#         ).eval()
#         tuned = PeftModel.from_pretrained(base, args.out_dir).eval()

#         def gen(sample_text: str) -> str:
#             ids = tokenizer(sample_text, return_tensors="pt").to(tuned.device)
#             with torch.no_grad():
#                 out = tuned.generate(
#                     **ids,
#                     max_new_tokens=128,
#                     do_sample=False,
#                     temperature=0.0,
#                     pad_token_id=tokenizer.eos_token_id,
#                 )
#             decoded = tokenizer.decode(out[0], skip_special_tokens=True)
#             # return only the part after RESP_HDR if present
#             return decoded.split(RESP_HDR, 1)[-1].strip()

#         print("\n[demo] Generation samples:")
#         for rec in val_ds.select(range(min(3, len(val_ds)))):
#             text = rec["text"]
#             prompt = text.split(RESP_HDR, 1)[0] + RESP_HDR
#             pred = gen(prompt)
#             print("----")
#             print(pred)

#     except Exception as e:
#         print(f"[warn] Demo generation skipped: {e}")

# if __name__ == "__main__":
#     torch.backends.cuda.matmul.allow_tf32 = True
#     main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, math, json, re
from typing import Dict, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    default_data_collator
)
from peft import LoraConfig, get_peft_model

# Optional TRL (for response-only masking)
try:
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
    HAS_TRL = True
except Exception:
    HAS_TRL = False

INSTR_HDR = "### Instruction:\n"
INPUT_HDR = "### Input:\n"
RESP_HDR  = "### Response:\n"

ALLOWED_STANCES = {"for": "For", "against": "Against", "neutral": "Neutral", "unrelated": "Unrelated"}

def normalize_header(name: str) -> str:
    n = name.strip().strip(",").lower()
    n = re.sub(r"\s+", "_", n)
    return n

def normalize_stance(x: str) -> str:
    if x is None:
        return "Unrelated"
    s = str(x).strip().lower().replace(".", "")
    s = s.replace("favour", "for").replace("in_favor", "for").replace("in favour", "for")
    s = s.replace("oppose", "against")
    return ALLOWED_STANCES.get(s, "Unrelated")

def build_prompt_and_response(row: Dict) -> Tuple[str, str]:
    tweet   = str(row.get("tweet", "")).strip()
    keyword = str(row.get("keyword", "")).strip()
    stance  = normalize_stance(row.get("stance", ""))
    reason  = str(row.get("stance_reason", "")).strip()

    if not reason:
        reason = {
            "Neutral": "No clear support or opposition is expressed.",
            "For":     "Expresses support or positive sentiment.",
            "Against": "Expresses opposition or negative sentiment.",
        }.get(stance, "No stance toward the target is expressed.")

    instruction = (
        'Given a tweet and a target keyword, classify the tweet’s stance toward the target as one of '
        '"For", "Against", "Neutral", or "Unrelated" and provide a concise reason grounded in the tweet. '
        'Return ONLY a compact JSON object with keys "stance" and "reason" (no extra text).'
    )
    model_input = f"target: {keyword}\n\ntweet: {tweet}"
    gold_json = json.dumps({"stance": stance, "reason": reason}, ensure_ascii=False)
    prompt = f"{INSTR_HDR}{instruction}\n\n{INPUT_HDR}{model_input}\n\n{RESP_HDR}"
    return prompt, gold_json

def load_and_prepare_csv(csv_path: str):
    ds = load_dataset("csv", data_files=csv_path)["train"]
    
    # First, find the columns we need (case-insensitive, before normalization)
    # This avoids duplicate column errors when multiple columns normalize to the same name
    cols = ds.column_names
    
    def find_col(candidates):
        """Find first matching column from candidates (case-insensitive)."""
        for cand in candidates:
            for c in cols:
                if normalize_header(c) == cand:
                    return c
        return None
    
    tweet_col = find_col(["tweet", "text", "content"])
    keyword_col = find_col(["keyword", "entity", "topic"])
    stance_col = find_col(["stance", "stance_label", "label"])
    reason_col = find_col(["stance_reason", "reason", "reasoning"])
    
    if not tweet_col:
        raise ValueError(f"Required column 'tweet' not found. Got: {cols}")
    if not keyword_col:
        raise ValueError(f"Required column 'keyword' not found. Got: {cols}")
    if not stance_col:
        raise ValueError(f"Required column 'stance' not found. Got: {cols}")
    
    # Keep only the columns we need (avoids duplicate rename issues)
    keep_cols = [c for c in [tweet_col, keyword_col, stance_col, reason_col] if c]
    ds = ds.select_columns(keep_cols)
    
    # Now rename to normalized names
    rename_map = {tweet_col: "tweet", keyword_col: "keyword", stance_col: "stance"}
    if reason_col:
        rename_map[reason_col] = "stance_reason"
    ds = ds.rename_columns(rename_map)

    if "stance_reason" not in ds.column_names:
        ds = ds.add_column("stance_reason", [""] * len(ds))

    def to_text_examples(batch):
        texts = []
        for t, k, s, r in zip(batch["tweet"], batch["keyword"], batch["stance"], batch["stance_reason"]):
            prompt, target = build_prompt_and_response(
                {"tweet": t, "keyword": k, "stance": s, "stance_reason": r}
            )
            texts.append(prompt + target)
        return {"text": texts}

    ds = ds.map(to_text_examples, batched=True, remove_columns=ds.column_names)
    return ds

def make_tokenizer(model_dir: str):
    try:
        tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok

def manual_tokenize_with_mask(dataset, tokenizer, max_length: int):
    def tokenize(batch):
        input_texts = batch["text"]
        enc = tokenizer(
            input_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        labels = enc["input_ids"].clone()

        prompts = []
        for text in input_texts:
            prompts.append(text.split(RESP_HDR, 1)[0] + RESP_HDR if RESP_HDR in text else text)

        prompt_ids = tokenizer(
            prompts, padding=False, truncation=True, max_length=max_length, add_special_tokens=True
        )["input_ids"]

        for i, p_ids in enumerate(prompt_ids):
            labels[i, :len(p_ids)] = -100

        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch")
    return tokenized

def get_lora_config():
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj","k_proj","v_proj","o_proj",
            "gate_proj","up_proj","down_proj"
        ],
    )

def load_fp_model_with_lora(model_dir: str, bf16: bool):
    # We want to set compute dtype (fp16 or bf16) when loading model, but
    # MistralForCausalLM doesn’t accept `dtype` param directly in its constructor.
    # Instead, use the `torch_dtype` argument of from_pretrained (if supported),
    # or fall back to loading default and casting later.
    torch_dtype = torch.bfloat16 if bf16 else torch.float16

    # Try passing torch_dtype to from_pretrained
    try:
        base = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    except TypeError:
        # Fallback: load without dtype argument, then convert
        base = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            trust_remote_code=True,
        )
        # Then cast to desired dtype
        base = base.to(torch_dtype)

    base.gradient_checkpointing_enable()
    base.config.use_cache = False

    lora_cfg = get_lora_config()
    model = get_peft_model(base, lora_cfg)
    return model


@torch.no_grad()
def eval_perplexity_response_only(model, tokenizer, raw_ds, max_length=1024, batch_size=4):
    tmp = manual_tokenize_with_mask(raw_ds, tokenizer, max_length)
    loader = torch.utils.data.DataLoader(tmp, batch_size=batch_size, collate_fn=default_data_collator)
    losses = []
    device = next(model.parameters()).device
    model.eval()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        losses.append(model(**batch).loss.item())
    return math.exp(sum(losses) / max(1, len(losses)))

def build_training_args(args):
    """Build TrainingArguments and gracefully handle older HF versions."""
    common = dict(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=25,
        report_to="none",
        remove_unused_columns=False,
        bf16=args.bf16,
        fp16=(not args.bf16),
    )
    # Try modern API first
    try:
        return TrainingArguments(save_strategy="epoch", evaluation_strategy="epoch", **common)
    except TypeError:
        # Fallback for older Transformers that don't know evaluation_strategy
        try:
            return TrainingArguments(save_strategy="epoch", **common)
        except TypeError:
            # Oldest fallback (pre-save_strategy). Uses steps-only logging/saving.
            common.pop("remove_unused_columns", None)
            return TrainingArguments(**common)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--eval-ppl", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[info] Loading CSV: {args.data_csv}")
    ds = load_and_prepare_csv(args.data_csv)
    ds = ds.train_test_split(test_size=0.1, seed=42)
    train_ds, val_ds = ds["train"], ds["test"]

    print(f"[info] Loading tokenizer & model from: {args.model_dir}")
    tokenizer = make_tokenizer(args.model_dir)
    model = load_fp_model_with_lora(args.model_dir, bf16=args.bf16)

    training_args = build_training_args(args)

    if HAS_TRL:
        print("[info] Using TRL SFTTrainer with response-only masking")
        collator = DataCollatorForCompletionOnlyLM(response_template=RESP_HDR, tokenizer=tokenizer)
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            max_seq_length=args.max_length,
            packing=False,
            dataset_text_field="text",
            data_collator=collator,
        )
    else:
        print("[info] TRL not found. Using manual masking + HF Trainer")
        train_tok = manual_tokenize_with_mask(train_ds, tokenizer, args.max_length)
        val_tok   = manual_tokenize_with_mask(val_ds, tokenizer, args.max_length)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
        )

    trainer.train()
    print("[info] Saving LoRA adapter to:", args.out_dir)
    trainer.model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    if args.eval_ppl:
        print("[info] Computing response-only perplexity on validation split …")
        ppl = eval_perplexity_response_only(trainer.model, tokenizer, val_ds, args.max_length, batch_size=4)
        print(f"[metric] Val PPL (response-only): {ppl:.3f}")

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
