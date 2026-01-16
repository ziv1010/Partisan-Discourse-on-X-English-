#!/usr/bin/env python3
# translate_eval_500.py  (works for any size; used for 5,000-run too)
import argparse, os, sys, html
from typing import List, Optional
import pandas as pd

def add_script_if_missing(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    if "devanagari_ratio" in df.columns and "script" in df.columns:
        return df
    def ratio(s: str) -> float:
        s = str(s) if s is not None else ""
        if not s: return 0.0
        return sum(1 for ch in s if '\u0900' <= ch <= '\u097F') / max(len(s), 1)
    out = df.copy()
    s = out[text_col].astype(str)
    out["devanagari_ratio"] = s.apply(ratio)
    out["script"] = out["devanagari_ratio"].map(lambda x: "devanagari" if x >= 0.4 else "roman_hindi")
    return out

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def ensure_transformers():
    try:
        import transformers, torch, sentencepiece  # noqa
    except Exception:
        sys.exit("Install offline deps: pip install transformers torch sentencepiece")

def ensure_gcloud():
    try:
        from google.cloud import translate_v2 as translate  # noqa
    except Exception:
        sys.exit("Install Google Translate client: pip install google-cloud-translate==2.*")

def translate_nllb(texts, max_new_tokens=128, batch_size=16, model_id="facebook/nllb-200-distilled-600M"):
    ensure_transformers()
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
    try:
        tok.src_lang = "hin_Deva"
        bos = tok.lang_code_to_id["eng_Latn"]
    except Exception:
        bos = None
    outs = []
    for batch in chunks(texts, batch_size):
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        if bos is not None:
            gen = model.generate(**enc, forced_bos_token_id=bos, max_new_tokens=max_new_tokens)
        else:
            gen = model.generate(**enc, max_new_tokens=max_new_tokens)
        outs.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return outs

def translate_opus(texts, max_new_tokens=128, batch_size=32, model_id="Helsinki-NLP/opus-mt-hi-en"):
    ensure_transformers()
    import torch
    from transformers import MarianTokenizer, MarianMTModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = MarianTokenizer.from_pretrained(model_id)
    model = MarianMTModel.from_pretrained(model_id).to(device)
    outs = []
    for batch in chunks(texts, batch_size):
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens)
        outs.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return outs

def translate_gcloud(texts, project=None, batch_size=80):
    ensure_gcloud()
    from google.cloud import translate_v2 as translate
    client = translate.Client(project=project) if project else translate.Client()
    outs = []
    for batch in chunks(texts, batch_size):
        res = client.translate(batch, target_language="en")
        outs.extend([html.unescape(r.get("translatedText", "")) for r in res])
    return outs

def main():
    ap = argparse.ArgumentParser(description="Translate Hindi tweets to English (any sample size).")
    ap.add_argument("--input", required=True)
    ap.add_argument("--text-col", default="tweet")
    ap.add_argument("--engine", choices=["nllb", "opus", "gcloud"], required=True)
    ap.add_argument("--out", default="eval_5000_en.csv")
    ap.add_argument("--project", default=None)
    ap.add_argument("--only-devanagari", action="store_true")
    ap.add_argument("--cache", default="translation_cache.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if args.text_col not in df.columns:
        sys.exit(f"Missing text column '{args.text_col}'. Available: {list(df.columns)}")
    df = add_script_if_missing(df, args.text_col)

    cache = {}
    if os.path.exists(args.cache):
        try:
            cdf = pd.read_csv(args.cache)
            if {"text","en_text"}.issubset(cdf.columns):
                cache = dict(zip(cdf["text"].astype(str), cdf["en_text"].astype(str)))
        except Exception:
            pass

    to_idx, to_txt = [], []
    for i, row in df.iterrows():
        text = str(row[args.text_col])
        if args.only_devanagari and str(row.get("script","")) != "devanagari":
            continue
        if text in cache:
            continue
        to_idx.append(i); to_txt.append(text)

    print(f"[info] Rows needing translation: {len(to_txt)}")

    if to_txt:
        if args.engine == "nllb":
            outs = translate_nllb(to_txt)
        elif args.engine == "opus":
            outs = translate_opus(to_txt)
        else:
            outs = translate_gcloud(to_txt, project=args.project)
        for i, en in zip(to_idx, outs):
            cache[str(df.at[i, args.text_col])] = en

    en_col = []
    for i, row in df.iterrows():
        t = str(row[args.text_col])
        en_col.append(cache.get(t, t if (args.only_devanagari and row.get("script")!="devanagari") else ""))
    df["en_text"] = en_col
    df["translation_engine"] = args.engine

    df.to_csv(args.out, index=False)
    pd.DataFrame({"text": list(cache.keys()), "en_text": list(cache.values())}).to_csv(args.cache, index=False)

    covered = sum(1 for t in df[args.text_col].astype(str) if t in cache)
    print(f"[OK] Wrote {args.out}. Translated {covered}/{len(df)} rows with {args.engine}. Cache: {args.cache}")

if __name__ == "__main__":
    main()
