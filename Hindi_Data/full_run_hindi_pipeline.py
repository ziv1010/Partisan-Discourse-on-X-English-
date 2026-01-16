#!/usr/bin/env python3
"""
Full Hindi pipeline runner (chunked, CPU-friendly):
 1) Translate Hindi/Roman Hindi tweets to English in chunks (NLLB or Opus).
 2) Embed originals + translations (paraphrase-multilingual-MiniLM-L12-v2) per chunk.
 3) Compute cosine similarity; write a translated CSV with cosine per row.
 4) Derive QC thresholds from empirical quantiles (or user-provided) and emit low-cos rows.

Example:
  MPLBACKEND=Agg MPLCONFIGDIR=/tmp ./pdxvenv/bin/python3 full_run_hindi_pipeline.py \
  --inputs-glob "hindi_tweets_essential_part*.csv" \
  --text-col tweet \
  --engine nllb \
  --out-translated full_hi_en.csv \
  --chunk-size 5000 \
  --batch-size 16 \
  --inspect-quantile 0.10 \
  --highrisk-quantile 0.05 \
  --lowcos-out lowcos_full.csv
"""
import argparse
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    MarianMTModel,
    MarianTokenizer,
)

import translate_eval_500 as tr


def _resolve_forced_bos(tok, code: str) -> int:
    """Best-effort lookup of BOS id for target lang code; returns None if unavailable."""
    if hasattr(tok, "lang_code_to_id") and tok.lang_code_to_id is not None:
        try:
            return tok.lang_code_to_id.get(code)
        except Exception:
            pass
    candidates = [code, f"<2{code}>", f"__{code}__", f"<<{code}>>"]
    for cand in candidates:
        try:
            tid = tok.convert_tokens_to_ids(cand)
        except Exception:
            tid = None
        if tid is not None and tid != getattr(tok, "unk_token_id", None):
            return tid
    return None


def make_translator(engine: str, device: str, max_new_tokens: int = 128, batch_size: int = 16) -> Callable[[List[str]], List[str]]:
    if engine == "nllb":
        tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="hin_Deva")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(device)
        bos = _resolve_forced_bos(tok, "eng_Latn")

        def _translate(texts: List[str]) -> List[str]:
            outs = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = tok(batch, return_tensors="pt", padding=True, truncation=True).to(device)
                gen = model.generate(**enc, forced_bos_token_id=bos, max_new_tokens=max_new_tokens) if bos is not None else model.generate(**enc, max_new_tokens=max_new_tokens)
                outs.extend(tok.batch_decode(gen, skip_special_tokens=True))
            return outs

        return _translate

    if engine == "opus":
        tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-hi-en")
        model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-hi-en").to(device)

        def _translate(texts: List[str]) -> List[str]:
            outs = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = tok(batch, return_tensors="pt", padding=True, truncation=True).to(device)
                gen = model.generate(**enc, max_new_tokens=max_new_tokens)
                outs.extend(tok.batch_decode(gen, skip_special_tokens=True))
            return outs

        return _translate

    raise SystemExit(f"Unsupported engine: {engine}")


def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def summarize_cos(sims: np.ndarray, label: str) -> None:
    qs = np.quantile(sims, [0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0])
    print(f"[cosine] {label}: min={qs[0]:.4f} p10={qs[1]:.4f} p25={qs[2]:.4f} median={qs[3]:.4f} p75={qs[4]:.4f} p90={qs[5]:.4f} max={qs[6]:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data_scripts/hindi_tweets_essential.csv", help="Path to input CSV (Hindi/Roman Hindi only)")
    ap.add_argument("--inputs-glob", default=None, help="Optional glob (e.g., data_scripts/hindi_tweets_essential_part_*.csv) to process multiple shard files")
    ap.add_argument("--text-col", default="tweet", help="Text column name")
    ap.add_argument("--engine", choices=["nllb", "opus"], default="opus", help="Translation engine")
    ap.add_argument("--out-translated", default="full_hi_en.csv", help="Output CSV with en_text + cosine added")
    ap.add_argument("--chunk-size", type=int, default=5000, help="Rows per chunk for streaming")
    ap.add_argument("--batch-size", type=int, default=16, help="Batch size for translation and embedding")
    ap.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens for generation")
    ap.add_argument("--inspect-thr", type=float, default=None, help="Cosine threshold to flag for inspection (if unset, derived from empirical quantile)")
    ap.add_argument("--highrisk-thr", type=float, default=None, help="Cosine threshold to mark high risk (if unset, derived from empirical quantile)")
    ap.add_argument("--inspect-quantile", type=float, default=0.10, help="Quantile for auto inspect threshold when --inspect-thr is unset")
    ap.add_argument("--highrisk-quantile", type=float, default=0.05, help="Quantile for auto high-risk threshold when --highrisk-thr is unset")
    ap.add_argument("--lowcos-out", default="lowcos_full.csv", help="Output CSV of rows below inspect threshold")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    translator = make_translator(args.engine, device, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
    embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    sims_all: List[float] = []
    scripts_all: List[str] = []

    out_path = Path(args.out_translated)
    if out_path.exists():
        out_path.unlink()

    # Build list of input files
    if args.inputs_glob:
        import glob
        files = sorted(glob.glob(args.inputs_glob))
        if not files:
            raise SystemExit(f"No files matched glob: {args.inputs_glob}")
    else:
        files = [args.input]

    wrote_header = False
    total_rows = 0
    chunk_global_idx = 0

    for path in files:
        chunk_iter = pd.read_csv(path, chunksize=args.chunk_size)
        for chunk_idx, chunk in enumerate(chunk_iter):
            if args.text_col not in chunk.columns:
                raise SystemExit(f"Missing text column '{args.text_col}' in chunk {chunk_idx} file {path}")
            chunk = tr.add_script_if_missing(chunk, args.text_col)

            texts = chunk[args.text_col].astype(str).tolist()
            en_texts = translator(texts)

            orig_emb = embed_texts(embedder, texts, batch_size=args.batch_size)
            en_emb = embed_texts(embedder, en_texts, batch_size=args.batch_size)
            sims = np.sum(orig_emb * en_emb, axis=1)

            chunk = chunk.copy()
            chunk["en_text"] = en_texts
            chunk["translation_engine"] = args.engine
            chunk["cosine_before_after"] = sims

            chunk.to_csv(out_path, mode="a", header=not wrote_header, index=False)
            wrote_header = True

            sims_all.extend(sims.tolist())
            if "script" in chunk.columns:
                scripts_all.extend(chunk["script"].astype(str).tolist())
            total_rows += len(chunk)
            print(f"[file {path}] chunk {chunk_global_idx} processed {len(chunk)} rows; total {total_rows}")
            chunk_global_idx += 1

    sims_all_np = np.array(sims_all, dtype=np.float32)
    summarize_cos(sims_all_np, label="overall")
    if scripts_all:
        scripts_np = np.array(scripts_all)
        for s in np.unique(scripts_np):
            summarize_cos(sims_all_np[scripts_np == s], label=f"script={s}")

    # Determine thresholds
    if args.inspect_thr is None:
        inspect_thr = float(np.quantile(sims_all_np, args.inspect_quantile))
        print(f"[qc] auto inspect threshold from quantile {args.inspect_quantile:.2f}: {inspect_thr:.4f}")
    else:
        inspect_thr = args.inspect_thr
        print(f"[qc] inspect threshold provided: {inspect_thr:.4f}")

    if args.highrisk_thr is None:
        highrisk_thr = float(np.quantile(sims_all_np, args.highrisk_quantile))
        print(f"[qc] auto high-risk threshold from quantile {args.highrisk_quantile:.2f}: {highrisk_thr:.4f}")
    else:
        highrisk_thr = args.highrisk_thr
        print(f"[qc] high-risk threshold provided: {highrisk_thr:.4f}")

    print(f"[qc] counts: <inspect {np.sum(sims_all_np < inspect_thr)} / {len(sims_all_np)}, <high-risk {np.sum(sims_all_np < highrisk_thr)} / {len(sims_all_np)}")

    # Emit low-cos rows in a second streaming pass over the translated file
    lowcos_path = Path(args.lowcos_out)
    if lowcos_path.exists():
        lowcos_path.unlink()

    wrote_low_header = False
    for chunk in pd.read_csv(out_path, chunksize=args.chunk_size):
        mask = chunk["cosine_before_after"] < inspect_thr
        if mask.any():
            low = chunk.loc[mask, :]
            low.to_csv(lowcos_path, mode="a", header=not wrote_low_header, index=False)
            wrote_low_header = True

    print(f"[qc] low-cos rows (<{inspect_thr:.4f}) written to {lowcos_path}")


if __name__ == "__main__":
    main()
