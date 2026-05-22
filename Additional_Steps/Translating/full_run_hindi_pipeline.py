#!/usr/bin/env python3
"""
Chunked Hindi/Indic-to-English translation runner with optional per-row NLLB source
language routing.

This keeps the original workflow intact:
1. Translate text to English in chunks.
2. Optionally embed originals + translations.
3. Compute cosine similarity per row.
4. Optionally emit low-cos rows for inspection.
"""

import argparse
import glob
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Dict, List, Optional

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


NLLB_LANG_ALIASES: Dict[str, str] = {
    "asm_Beng": "asm_Beng",
    "ben_Beng": "ben_Beng",
    "brx_Deva": "brx_Deva",
    "doi_Deva": "doi_Deva",
    "eng_Latn": "eng_Latn",
    "gom_Deva": "gom_Deva",
    "guj_Gujr": "guj_Gujr",
    "hin_Deva": "hin_Deva",
    "kan_Knda": "kan_Knda",
    "kas_Arab": "kas_Arab",
    "kas_Deva": "kas_Deva",
    "kok_Deva": "gom_Deva",
    "mai_Deva": "mai_Deva",
    "mal_Mlym": "mal_Mlym",
    "mar_Deva": "mar_Deva",
    "mni_Beng": "mni_Beng",
    "mni_Meti": "mni_Mtei",
    "mni_Mtei": "mni_Mtei",
    "nep_Deva": "npi_Deva",
    "npi_Deva": "npi_Deva",
    "ori_Orya": "ory_Orya",
    "ory_Orya": "ory_Orya",
    "pan_Guru": "pan_Guru",
    "san_Deva": "san_Deva",
    "sat_Olch": "sat_Olck",
    "sat_Olck": "sat_Olck",
    "snd_Arab": "snd_Arab",
    "snd_Deva": "snd_Deva",
    "tam_Tamil": "tam_Taml",
    "tam_Taml": "tam_Taml",
    "tel_Telu": "tel_Telu",
    "urd_Arab": "urd_Arab",
    "hin_Latn": "hin_Deva",
    "mar_Latn": "mar_Deva",
    "nep_Latn": "npi_Deva",
    "mai_Latn": "mai_Deva",
    "kok_Latn": "gom_Deva",
    "asm_Latn": "asm_Beng",
    "ben_Latn": "ben_Beng",
    "brx_Latn": "brx_Deva",
    "guj_Latn": "guj_Gujr",
    "kan_Latn": "kan_Knda",
    "kas_Latn": "kas_Arab",
    "mal_Latn": "mal_Mlym",
    "mni_Latn": "mni_Beng",
    "ori_Latn": "ory_Orya",
    "pan_Latn": "pan_Guru",
    "san_Latn": "san_Deva",
    "snd_Latn": "snd_Arab",
    "tam_Latn": "tam_Taml",
    "tel_Latn": "tel_Telu",
    "urd_Latn": "urd_Arab",
}


def add_script_if_missing(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    if "devanagari_ratio" in df.columns and "script" in df.columns:
        return df

    def ratio(text: str) -> float:
        value = str(text) if text is not None else ""
        if not value:
            return 0.0
        devanagari = sum(1 for ch in value if "\u0900" <= ch <= "\u097F")
        return devanagari / max(len(value), 1)

    out = df.copy()
    series = out[text_col].astype(str)
    out["devanagari_ratio"] = series.apply(ratio)
    out["script"] = out["devanagari_ratio"].map(
        lambda value: "devanagari" if value >= 0.4 else "roman_hindi"
    )
    return out


def _resolve_forced_bos(tok, code: str) -> Optional[int]:
    if hasattr(tok, "lang_code_to_id") and tok.lang_code_to_id is not None:
        try:
            return tok.lang_code_to_id.get(code)
        except Exception:
            pass
    for candidate in (code, f"<2{code}>", f"__{code}__", f"<<{code}>>"):
        try:
            token_id = tok.convert_tokens_to_ids(candidate)
        except Exception:
            token_id = None
        if token_id is not None and token_id != getattr(tok, "unk_token_id", None):
            return token_id
    return None


def _normalize_nllb_src_lang(label: Optional[str], tok, default: str) -> str:
    raw = str(label or "").strip()
    candidate = NLLB_LANG_ALIASES.get(raw, raw or default)
    lang_codes = getattr(tok, "lang_code_to_id", {}) or {}
    if candidate in lang_codes:
        return candidate
    fallback = NLLB_LANG_ALIASES.get(candidate, default)
    if fallback in lang_codes:
        return fallback
    if default in lang_codes:
        return default
    return "hin_Deva"


def _resolve_torch_dtype(dtype_name: str, device: str):
    if device != "cuda":
        return None
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name == "auto":
        return torch.bfloat16
    return mapping[dtype_name]


class NLLBTranslator:
    def __init__(
        self,
        device: str,
        max_new_tokens: int,
        batch_size: int,
        default_src_lang: str,
        dtype_name: str,
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.default_src_lang = default_src_lang
        self.torch_dtype = _resolve_torch_dtype(dtype_name, device)
        self.tok = AutoTokenizer.from_pretrained(
            "facebook/nllb-200-distilled-600M",
            src_lang=default_src_lang,
        )
        load_kwargs = {
            "use_safetensors": True,
            "low_cpu_mem_usage": False,
        }
        if self.torch_dtype is not None:
            load_kwargs["torch_dtype"] = self.torch_dtype
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/nllb-200-distilled-600M",
            **load_kwargs,
        )
        self.model.to(self.device)
        self.model.eval()
        self.bos = _resolve_forced_bos(self.tok, "eng_Latn")

    def translate(self, texts: List[str], src_langs: Optional[List[str]] = None) -> List[str]:
        if src_langs is None:
            src_langs = [self.default_src_lang] * len(texts)
        outputs = [""] * len(texts)
        groups: Dict[str, List[int]] = {}
        for idx, src_lang in enumerate(src_langs):
            normalized = _normalize_nllb_src_lang(src_lang, self.tok, self.default_src_lang)
            groups.setdefault(normalized, []).append(idx)

        for src_lang, indices in groups.items():
            self.tok.src_lang = src_lang
            for start in range(0, len(indices), self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                batch = [texts[idx] for idx in batch_indices]
                enc = self.tok(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.device)
                autocast_ctx = nullcontext()
                if self.device == "cuda" and self.torch_dtype in {torch.float16, torch.bfloat16}:
                    autocast_ctx = torch.autocast(device_type="cuda", dtype=self.torch_dtype)
                with autocast_ctx:
                    if self.bos is not None:
                        generated = self.model.generate(
                            **enc,
                            forced_bos_token_id=self.bos,
                            max_new_tokens=self.max_new_tokens,
                        )
                    else:
                        generated = self.model.generate(
                            **enc,
                            max_new_tokens=self.max_new_tokens,
                        )
                decoded = self.tok.batch_decode(generated, skip_special_tokens=True)
                for row_idx, translated in zip(batch_indices, decoded):
                    outputs[row_idx] = translated
        return outputs


class OpusTranslator:
    def __init__(self, device: str, max_new_tokens: int, batch_size: int, dtype_name: str):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.torch_dtype = _resolve_torch_dtype(dtype_name, device)
        self.tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-hi-en")
        load_kwargs = {}
        if self.torch_dtype is not None:
            load_kwargs["torch_dtype"] = self.torch_dtype
        self.model = MarianMTModel.from_pretrained(
            "Helsinki-NLP/opus-mt-hi-en", **load_kwargs
        ).to(device)

    def translate(self, texts: List[str], src_langs: Optional[List[str]] = None) -> List[str]:
        if src_langs:
            distinct = {
                str(lang).strip()
                for lang in src_langs
                if str(lang).strip() and str(lang).strip() != "hin_Deva"
            }
            if distinct:
                raise SystemExit(
                    "Opus translator only supports Hindi source text. "
                    f"Found unsupported source languages: {sorted(distinct)}"
                )
        outputs: List[str] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            enc = self.tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            generated = self.model.generate(**enc, max_new_tokens=self.max_new_tokens)
            outputs.extend(self.tok.batch_decode(generated, skip_special_tokens=True))
        return outputs


def make_translator(
    engine: str,
    device: str,
    max_new_tokens: int,
    batch_size: int,
    default_src_lang: str,
    dtype_name: str,
) -> Callable[[List[str], Optional[List[str]]], List[str]]:
    if engine == "nllb":
        return NLLBTranslator(
            device=device,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            default_src_lang=default_src_lang,
            dtype_name=dtype_name,
        ).translate
    if engine == "opus":
        return OpusTranslator(
            device=device,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            dtype_name=dtype_name,
        ).translate
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
    print(
        "[cosine] "
        f"{label}: min={qs[0]:.4f} p10={qs[1]:.4f} p25={qs[2]:.4f} "
        f"median={qs[3]:.4f} p75={qs[4]:.4f} p90={qs[5]:.4f} max={qs[6]:.4f}"
    )


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for chunk in pd.read_csv(path, chunksize=100_000):
        total += len(chunk)
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data_scripts/hindi_tweets_essential.csv")
    ap.add_argument("--inputs-glob", default=None)
    ap.add_argument("--text-col", default="tweet")
    ap.add_argument("--src-lang", default="hin_Deva")
    ap.add_argument("--src-lang-col", default=None)
    ap.add_argument("--engine", choices=["nllb", "opus"], default="opus")
    ap.add_argument("--out-translated", default="full_hi_en.csv")
    ap.add_argument("--chunk-size", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--inspect-thr", type=float, default=None)
    ap.add_argument("--highrisk-thr", type=float, default=None)
    ap.add_argument("--inspect-quantile", type=float, default=0.10)
    ap.add_argument("--highrisk-quantile", type=float, default=0.05)
    ap.add_argument("--lowcos-out", default="lowcos_full.csv")
    ap.add_argument("--skip-embeddings", action="store_true")
    ap.add_argument("--skip-lowcos", action="store_true")
    ap.add_argument("--resume-if-exists", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    translator = make_translator(
        args.engine,
        device=device,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        default_src_lang=args.src_lang,
        dtype_name=args.dtype,
    )
    embedder = None if args.skip_embeddings else SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    out_path = Path(args.out_translated)
    resume_rows = 0
    if args.resume_if_exists and out_path.exists():
        resume_rows = _count_csv_rows(out_path)
        print(f"[resume] existing output detected at {out_path}; resuming after {resume_rows} rows")
    elif out_path.exists():
        out_path.unlink()

    if args.inputs_glob:
        files = sorted(glob.glob(args.inputs_glob))
        if not files:
            raise SystemExit(f"No files matched glob: {args.inputs_glob}")
    else:
        files = [args.input]

    sims_all: List[float] = []
    scripts_all: List[str] = []
    wrote_header = out_path.exists() and out_path.stat().st_size > 0
    total_rows = resume_rows
    chunk_global_idx = total_rows // max(args.chunk_size, 1)

    if args.resume_if_exists and resume_rows > 0 and not args.skip_embeddings:
        print("[resume] loading prior cosine/script values from existing output")
        for existing_chunk in pd.read_csv(out_path, chunksize=args.chunk_size):
            if "cosine_before_after" in existing_chunk.columns:
                sims = existing_chunk["cosine_before_after"].to_numpy(dtype=np.float32, copy=False)
                sims_all.extend(sims.tolist())
            if "script" in existing_chunk.columns:
                scripts_all.extend(existing_chunk["script"].astype(str).tolist())

    rows_to_skip = resume_rows

    for path in files:
        for chunk_idx, chunk in enumerate(pd.read_csv(path, chunksize=args.chunk_size)):
            if rows_to_skip >= len(chunk):
                rows_to_skip -= len(chunk)
                continue
            if rows_to_skip > 0:
                chunk = chunk.iloc[rows_to_skip:].copy()
                rows_to_skip = 0
            if args.text_col not in chunk.columns:
                raise SystemExit(
                    f"Missing text column '{args.text_col}' in chunk {chunk_idx} file {path}"
                )
            chunk = add_script_if_missing(chunk, args.text_col)
            texts = chunk[args.text_col].fillna("").astype(str).tolist()
            src_langs = None
            if args.src_lang_col:
                if args.src_lang_col not in chunk.columns:
                    raise SystemExit(
                        f"Missing source language column '{args.src_lang_col}' in file {path}"
                    )
                src_langs = chunk[args.src_lang_col].fillna(args.src_lang).astype(str).tolist()

            en_texts = translator(texts, src_langs)

            if args.skip_embeddings:
                sims = np.full(len(texts), np.nan, dtype=np.float32)
            else:
                assert embedder is not None
                orig_emb = embed_texts(embedder, texts, batch_size=args.batch_size)
                en_emb = embed_texts(embedder, en_texts, batch_size=args.batch_size)
                sims = np.sum(orig_emb * en_emb, axis=1)

            chunk = chunk.copy()
            chunk["en_text"] = en_texts
            chunk["translation_engine"] = args.engine
            chunk["cosine_before_after"] = sims
            chunk.to_csv(out_path, mode="a", header=not wrote_header, index=False)
            wrote_header = True

            if not args.skip_embeddings:
                sims_all.extend(sims.tolist())
                if "script" in chunk.columns:
                    scripts_all.extend(chunk["script"].astype(str).tolist())
            total_rows += len(chunk)
            print(f"[file {path}] chunk {chunk_global_idx} processed {len(chunk)} rows; total {total_rows}")
            chunk_global_idx += 1

    if args.skip_embeddings:
        print("[qc] Embeddings skipped; cosine summaries and low-cos export were not computed.")
        return

    sims_all_np = np.array(sims_all, dtype=np.float32)
    summarize_cos(sims_all_np, label="overall")
    if scripts_all:
        scripts_np = np.array(scripts_all)
        for script in np.unique(scripts_np):
            summarize_cos(sims_all_np[scripts_np == script], label=f"script={script}")

    if args.inspect_thr is None:
        inspect_thr = float(np.quantile(sims_all_np, args.inspect_quantile))
        print(f"[qc] auto inspect threshold from quantile {args.inspect_quantile:.2f}: {inspect_thr:.4f}")
    else:
        inspect_thr = args.inspect_thr
        print(f"[qc] inspect threshold provided: {inspect_thr:.4f}")

    if args.highrisk_thr is None:
        highrisk_thr = float(np.quantile(sims_all_np, args.highrisk_quantile))
        print(
            f"[qc] auto high-risk threshold from quantile {args.highrisk_quantile:.2f}: "
            f"{highrisk_thr:.4f}"
        )
    else:
        highrisk_thr = args.highrisk_thr
        print(f"[qc] high-risk threshold provided: {highrisk_thr:.4f}")

    print(
        f"[qc] counts: <inspect {np.sum(sims_all_np < inspect_thr)} / {len(sims_all_np)}, "
        f"<high-risk {np.sum(sims_all_np < highrisk_thr)} / {len(sims_all_np)}"
    )

    if args.skip_lowcos:
        print("[qc] Low-cos export skipped.")
        return

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
