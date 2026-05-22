#!/usr/bin/env python3
"""
Split a large tweet CSV into English, native-script Indic, and Roman Hindi/Hinglish
branches before translation.

Outputs:
  - english_tweets.csv
  - hindi_script_tweets.csv
  - hinglish_roman_tweets.csv
  - routing_summary.csv

The Roman branch is normalized to native script with selective transliteration so
common English words, URLs, mentions, and similar tokens remain untouched.
"""

import argparse
import csv
import re
import shutil
import urllib.request
import zipfile
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import fasttext
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from wordfreq import zipf_frequency

from ai4bharat.transliteration import XlitEngine


INDICLID_MODEL_URLS = {
    "indiclid-ftn": "https://github.com/AI4Bharat/IndicLID/releases/download/v1.0/indiclid-ftn.zip",
    "indiclid-ftr": "https://github.com/AI4Bharat/IndicLID/releases/download/v1.0/indiclid-ftr.zip",
    "indiclid-bert": "https://github.com/AI4Bharat/IndicLID/releases/download/v1.0/indiclid-bert.zip",
}

XLIT_LANG_MAP = {
    "asm": "as",
    "ben": "bn",
    "brx": "brx",
    "guj": "gu",
    "hin": "hi",
    "kan": "kn",
    "kas": "ks",
    "kok": "gom",
    "mai": "mai",
    "mal": "ml",
    "mni": "mni",
    "mar": "mr",
    "nep": "ne",
    "ori": "or",
    "ory": "or",
    "pan": "pa",
    "san": "sa",
    "snd": "sd",
    "tam": "ta",
    "tel": "te",
    "urd": "ur",
}

ROMAN_TO_NATIVE_LABEL = {
    "asm_Latn": "asm_Beng",
    "ben_Latn": "ben_Beng",
    "brx_Latn": "brx_Deva",
    "guj_Latn": "guj_Gujr",
    "hin_Latn": "hin_Deva",
    "kan_Latn": "kan_Knda",
    "kas_Latn": "kas_Arab",
    "kok_Latn": "kok_Deva",
    "mai_Latn": "mai_Deva",
    "mal_Latn": "mal_Mlym",
    "mni_Latn": "mni_Beng",
    "mar_Latn": "mar_Deva",
    "nep_Latn": "nep_Deva",
    "ori_Latn": "ori_Orya",
    "pan_Latn": "pan_Guru",
    "san_Latn": "san_Deva",
    "snd_Latn": "snd_Arab",
    "tam_Latn": "tam_Tamil",
    "tel_Latn": "tel_Telu",
    "urd_Latn": "urd_Arab",
}

TOKEN_RE = re.compile(
    r"https?://\S+|www\.\S+|@\w+|#\w+|[A-Za-z]+(?:['’][A-Za-z]+)?|\d+|[^\w\s]+|\s+",
    re.UNICODE,
)
WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
URL_RE = re.compile(r"^(?:https?://|www\.)", re.IGNORECASE)
INDIC_SCRIPT_RE = re.compile(
    r"[\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F"
    r"\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F\u0600-\u06FF"
    r"\u1C50-\u1C7F\u1CDA-\u1CDF\uABC0-\uABFF]"
)

HINDI_ROMAN_KEEP_AS_HINDI = {
    "aaj", "ab", "acha", "achha", "agar", "aisa", "aisi", "aise", "apna", "apni",
    "apne", "aur", "bahut", "bhi", "bhai", "bola", "chalo", "dekh", "ek", "fir",
    "galat", "hai", "hain", "ham", "hame", "hamesha", "hum", "is", "jab", "jana",
    "jo", "ka", "kah", "kaha", "kahin", "kar", "karo", "ke", "ki", "koi", "ko",
    "kya", "kyu", "kyon", "le", "liye", "main", "mai", "mera", "meri", "mere",
    "mujhe", "na", "nahi", "nahin", "par", "pe", "sab", "se", "sirf", "tha", "thi",
    "the", "to", "tum", "unka", "unki", "unke", "us", "wah", "wo", "woh", "ya",
    "ye", "yah",
}


class TextDataset(Dataset):
    def __init__(self, rows: List[Tuple[int, str]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[int, str]:
        return self.rows[idx]


class IndicLIDModel:
    def __init__(
        self,
        model_root: Path,
        input_threshold: float = 0.5,
        roman_lid_threshold: float = 0.6,
        device: Optional[str] = None,
    ):
        self.model_root = model_root
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.input_threshold = input_threshold
        self.model_threshold = roman_lid_threshold

        self.ft_native = fasttext.load_model(
            str(model_root / "indiclid-ftn" / "model_baseline_roman.bin")
        )
        self.ft_roman = fasttext.load_model(
            str(model_root / "indiclid-ftr" / "model_baseline_roman.bin")
        )
        self.roman_bert = torch.load(
            str(model_root / "indiclid-bert" / "basline_nn_simple.pt"),
            map_location=self.device,
            weights_only=False,
        )
        config = getattr(self.roman_bert, "config", None)
        if config is not None:
            if not hasattr(config, "_output_attentions"):
                config._output_attentions = getattr(config, "output_attentions", False)
            if not hasattr(config, "_output_hidden_states"):
                config._output_hidden_states = getattr(config, "output_hidden_states", False)
            if not hasattr(config, "_return_dict"):
                config._return_dict = getattr(config, "return_dict", True)
            if not hasattr(config, "_attn_implementation_internal"):
                config._attn_implementation_internal = "eager"
        if hasattr(self.roman_bert, "set_attn_implementation"):
            self.roman_bert.set_attn_implementation("eager")
        if hasattr(self.roman_bert, "bert") and not hasattr(self.roman_bert.bert, "attn_implementation"):
            self.roman_bert.bert.attn_implementation = "eager"
        self.roman_bert.eval()
        self.bert_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBERTv2-MLM-only")
        self.lang_code_dict_reverse = {
            0: "asm_Latn",
            1: "ben_Latn",
            2: "brx_Latn",
            3: "guj_Latn",
            4: "hin_Latn",
            5: "kan_Latn",
            6: "kas_Latn",
            7: "kok_Latn",
            8: "mai_Latn",
            9: "mal_Latn",
            10: "mni_Latn",
            11: "mar_Latn",
            12: "nep_Latn",
            13: "ori_Latn",
            14: "pan_Latn",
            15: "san_Latn",
            16: "snd_Latn",
            17: "tam_Latn",
            18: "tel_Latn",
            19: "urd_Latn",
            20: "eng_Latn",
            21: "other",
            22: "asm_Beng",
            23: "ben_Beng",
            24: "brx_Deva",
            25: "doi_Deva",
            26: "guj_Gujr",
            27: "hin_Deva",
            28: "kan_Knda",
            29: "kas_Arab",
            30: "kas_Deva",
            31: "kok_Deva",
            32: "mai_Deva",
            33: "mal_Mlym",
            34: "mni_Beng",
            35: "mni_Meti",
            36: "mar_Deva",
            37: "nep_Deva",
            38: "ori_Orya",
            39: "pan_Guru",
            40: "san_Deva",
            41: "sat_Olch",
            42: "snd_Arab",
            43: "tam_Tamil",
            44: "tel_Telu",
            45: "urd_Arab",
        }

    @staticmethod
    def char_percent_check(text: str) -> float:
        input_len = len(text)
        special_chars = len(re.findall(r"[@_!#$%^&*()<>?/\|}{~:]", text))
        spaces = len(re.findall(r"\s", text))
        total_chars = input_len - special_chars - spaces
        english_chars = len(re.findall(r"[a-zA-Z0-9]", text))
        if total_chars <= 0:
            return 0.0
        return english_chars / total_chars

    @staticmethod
    def normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", str(text)).strip()

    @staticmethod
    def contains_indic_script(text: str) -> bool:
        return bool(INDIC_SCRIPT_RE.search(text))

    def _native_inference(self, rows: List[Tuple[int, str]], output: Dict[int, Tuple[str, str, float, str]]):
        if not rows:
            return
        texts = [self.normalize_text(text) for _, text in rows]
        predictions = self.ft_native.predict(texts)
        for (idx, text), pred_label, pred_score in zip(rows, predictions[0], predictions[1]):
            output[idx] = (text, pred_label[0][9:], float(pred_score[0]), "IndicLID-FTN")

    def _roman_bert_inference(
        self,
        rows: List[Tuple[int, str]],
        output: Dict[int, Tuple[str, str, float, str]],
        batch_size: int,
    ):
        if not rows:
            return
        loader = DataLoader(TextDataset(rows), batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                batch_indices, batch_texts = batch
                batch_texts = [self.normalize_text(text) for text in batch_texts]
                enc = self.bert_tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                enc = {key: value.to(self.device) for key, value in enc.items()}
                model_out = self.roman_bert(
                    enc["input_ids"],
                    token_type_ids=enc["token_type_ids"],
                    attention_mask=enc["attention_mask"],
                )
                pred_indices = torch.argmax(model_out.logits, dim=1)
                for idx, text, pred_idx, logits in zip(batch_indices, batch_texts, pred_indices, model_out.logits):
                    label = self.lang_code_dict_reverse[int(pred_idx.item())]
                    score = float(logits[pred_idx].item())
                    output[int(idx)] = (text, label, score, "IndicLID-BERT")

    def _roman_inference(
        self,
        rows: List[Tuple[int, str]],
        output: Dict[int, Tuple[str, str, float, str]],
        batch_size: int,
    ):
        if not rows:
            return
        texts = [self.normalize_text(text) for _, text in rows]
        predictions = self.ft_roman.predict(texts)
        fallback_rows: List[Tuple[int, str]] = []
        for (idx, text), pred_label, pred_score in zip(rows, predictions[0], predictions[1]):
            score = float(pred_score[0])
            if score > self.model_threshold:
                output[idx] = (text, pred_label[0][9:], score, "IndicLID-FTR")
            else:
                fallback_rows.append((idx, text))
        self._roman_bert_inference(fallback_rows, output, batch_size=batch_size)

    def batch_predict(self, texts: List[str], batch_size: int) -> List[Tuple[str, str, float, str]]:
        output: Dict[int, Tuple[str, str, float, str]] = {}
        roman_rows: List[Tuple[int, str]] = []
        native_rows: List[Tuple[int, str]] = []
        for idx, text in enumerate(texts):
            if self.contains_indic_script(text):
                native_rows.append((idx, text))
            elif self.char_percent_check(text) > self.input_threshold:
                roman_rows.append((idx, text))
            else:
                native_rows.append((idx, text))
        self._native_inference(native_rows, output)
        self._roman_inference(roman_rows, output, batch_size=batch_size)
        return [output[idx] for idx in sorted(output)]


class TransliteratorPool:
    def __init__(self, beam_width: int, rescore: bool):
        self.beam_width = beam_width
        self.rescore = rescore
        self.engines: Dict[str, XlitEngine] = {}
        self.cache: Dict[Tuple[str, str], str] = {}

    def _get_engine(self, lang_code: str) -> XlitEngine:
        if lang_code not in self.engines:
            self.engines[lang_code] = XlitEngine(
                lang_code,
                beam_width=self.beam_width,
                rescore=self.rescore,
                src_script_type="roman",
            )
        return self.engines[lang_code]

    def translit_word(self, word: str, lang_code: str) -> str:
        key = (lang_code, word.lower())
        if key in self.cache:
            return self.cache[key]
        engine = self._get_engine(lang_code)
        result = engine.translit_word(word.lower(), lang_code=lang_code, topk=1)
        if isinstance(result, dict):
            values = result.get(lang_code, [])
            transliterated = values[0] if values else word
        elif isinstance(result, list):
            transliterated = result[0] if result else word
        else:
            transliterated = word
        self.cache[key] = transliterated
        return transliterated


def download_and_extract(url: str, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir.with_suffix(".zip")
    print(f"[download] {url}")
    with urllib.request.urlopen(url) as response, zip_path.open("wb") as out_file:
        shutil.copyfileobj(response, out_file)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dest_dir.parent)
    zip_path.unlink(missing_ok=True)


def ensure_indiclid_models(model_root: Path) -> None:
    model_root.mkdir(parents=True, exist_ok=True)
    required = {
        "indiclid-ftn": model_root / "indiclid-ftn" / "model_baseline_roman.bin",
        "indiclid-ftr": model_root / "indiclid-ftr" / "model_baseline_roman.bin",
        "indiclid-bert": model_root / "indiclid-bert" / "basline_nn_simple.pt",
    }
    for folder, required_file in required.items():
        if required_file.exists():
            continue
        download_and_extract(INDICLID_MODEL_URLS[folder], model_root / folder)
        if not required_file.exists():
            raise SystemExit(f"Missing required IndicLID artifact after download: {required_file}")


@lru_cache(maxsize=500_000)
def english_zipf(word: str) -> float:
    return float(zipf_frequency(word, "en"))


def should_preserve_english_token(token: str, english_zipf_threshold: float) -> bool:
    if not token or URL_RE.match(token) or token.startswith("@"):
        return True
    bare = token[1:] if token.startswith("#") else token
    if not bare:
        return True
    if not any(ch.isalpha() for ch in bare):
        return True
    lower = bare.lower()
    if lower in HINDI_ROMAN_KEEP_AS_HINDI:
        return False
    if bare.isupper() and len(bare) > 1:
        return True
    if len(lower) <= 1:
        return True
    return english_zipf(lower) >= english_zipf_threshold


def transliterate_mixed_text(
    text: str,
    lang_code: str,
    transliterators: TransliteratorPool,
    english_zipf_threshold: float,
) -> str:
    pieces: List[str] = []
    for token in TOKEN_RE.findall(text):
        if not token or token.isspace():
            pieces.append(token)
            continue
        if URL_RE.match(token) or token.startswith("@"):
            pieces.append(token)
            continue
        if token.startswith("#"):
            hashtag_body = token[1:]
            if WORD_RE.fullmatch(hashtag_body) and not should_preserve_english_token(token, english_zipf_threshold):
                pieces.append("#" + transliterators.translit_word(hashtag_body, lang_code))
            else:
                pieces.append(token)
            continue
        if WORD_RE.fullmatch(token):
            if should_preserve_english_token(token, english_zipf_threshold):
                pieces.append(token)
            else:
                pieces.append(transliterators.translit_word(token, lang_code))
            continue
        pieces.append(token)
    return "".join(pieces)


def xlit_lang_from_label(label: str) -> Optional[str]:
    prefix = label.split("_", 1)[0]
    return XLIT_LANG_MAP.get(prefix)


def route_for_label(label: str) -> str:
    if label == "eng_Latn" or label == "other":
        return "english"
    if label.endswith("_Latn"):
        return "hinglish_roman"
    return "hindi_script"


def write_chunk(df: pd.DataFrame, path: Path, wrote_header: bool) -> bool:
    df.to_csv(path, mode="a", header=not wrote_header, index=False, quoting=csv.QUOTE_MINIMAL)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--text-col", default="tweet")
    ap.add_argument("--output-dir", default=".")
    ap.add_argument("--chunk-size", type=int, default=100_000)
    ap.add_argument("--lid-batch-size", type=int, default=256)
    ap.add_argument("--beam-width", type=int, default=10)
    ap.add_argument("--english-zipf-threshold", type=float, default=4.2)
    ap.add_argument("--input-threshold", type=float, default=0.5)
    ap.add_argument("--roman-lid-threshold", type=float, default=0.6)
    ap.add_argument("--model-dir", default="models/indiclid")
    ap.add_argument("--roman-target-xlit-lang", default="hi")
    ap.add_argument("--roman-target-nllb-src", default="hin_Deva")
    ap.add_argument("--limit-rows", type=int, default=None)
    args = ap.parse_args()

    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = output_dir / model_dir

    english_path = output_dir / "english_tweets.csv"
    native_path = output_dir / "hindi_script_tweets.csv"
    roman_path = output_dir / "hinglish_roman_tweets.csv"
    summary_path = output_dir / "routing_summary.csv"
    for path in (english_path, native_path, roman_path, summary_path):
        path.unlink(missing_ok=True)

    ensure_indiclid_models(model_dir)
    lid = IndicLIDModel(
        model_root=model_dir,
        input_threshold=args.input_threshold,
        roman_lid_threshold=args.roman_lid_threshold,
    )
    transliterators = TransliteratorPool(beam_width=args.beam_width, rescore=True)

    total_rows = 0
    wrote_header = {"english": False, "hindi_script": False, "hinglish_roman": False}
    route_counts: Counter = Counter()
    lang_counts: Counter = Counter()

    for chunk_idx, chunk in enumerate(pd.read_csv(input_path, chunksize=args.chunk_size)):
        if args.limit_rows is not None and total_rows >= args.limit_rows:
            break
        if args.text_col not in chunk.columns:
            raise SystemExit(f"Missing text column '{args.text_col}' in {input_path}")

        if args.limit_rows is not None:
            remaining = args.limit_rows - total_rows
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining].copy()

        chunk = chunk.copy()
        texts = chunk[args.text_col].fillna("").astype(str).tolist()
        chunk["row_id"] = np.arange(total_rows, total_rows + len(chunk))
        chunk["tweet_original"] = texts

        results = lid.batch_predict(texts, batch_size=args.lid_batch_size)
        routes: List[str] = []
        labels: List[str] = []
        scores: List[float] = []
        models: List[str] = []
        xlit_langs: List[str] = []
        nllb_src_langs: List[str] = []
        normalized_texts: List[str] = []

        for text, label, score, model in results:
            route = route_for_label(label)
            xlit_lang = ""
            normalized_text = text
            nllb_src_lang = ""

            if route == "hinglish_roman":
                xlit_lang = args.roman_target_xlit_lang
                nllb_src_lang = args.roman_target_nllb_src
                normalized_text = transliterate_mixed_text(
                    text,
                    lang_code=xlit_lang,
                    transliterators=transliterators,
                    english_zipf_threshold=args.english_zipf_threshold,
                )
            elif route == "hindi_script":
                nllb_src_lang = label

            routes.append(route)
            labels.append(label)
            scores.append(score)
            models.append(model)
            xlit_langs.append(xlit_lang)
            nllb_src_langs.append(nllb_src_lang)
            normalized_texts.append(normalized_text)
            route_counts[route] += 1
            lang_counts[(route, label)] += 1

        chunk["translation_route"] = routes
        chunk["lid_lang"] = labels
        chunk["lid_score"] = scores
        chunk["lid_model"] = models
        chunk["xlit_lang"] = xlit_langs
        chunk["nllb_src_lang"] = nllb_src_langs
        chunk[args.text_col] = normalized_texts

        english_rows = chunk.loc[chunk["translation_route"] == "english"].copy()
        native_rows = chunk.loc[chunk["translation_route"] == "hindi_script"].copy()
        roman_rows = chunk.loc[chunk["translation_route"] == "hinglish_roman"].copy()

        if not english_rows.empty:
            wrote_header["english"] = write_chunk(english_rows, english_path, wrote_header["english"])
        if not native_rows.empty:
            wrote_header["hindi_script"] = write_chunk(native_rows, native_path, wrote_header["hindi_script"])
        if not roman_rows.empty:
            wrote_header["hinglish_roman"] = write_chunk(roman_rows, roman_path, wrote_header["hinglish_roman"])

        total_rows += len(chunk)
        print(
            f"[chunk {chunk_idx}] rows={len(chunk)} total={total_rows} "
            f"english={len(english_rows)} native={len(native_rows)} roman={len(roman_rows)}"
        )

    summary_rows = []
    for route, count in sorted(route_counts.items()):
        summary_rows.append({"level": "route", "route": route, "lid_lang": "", "count": count})
    for (route, label), count in sorted(lang_counts.items()):
        summary_rows.append({"level": "route_lang", "route": route, "lid_lang": label, "count": count})
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"[done] english -> {english_path}")
    print(f"[done] native  -> {native_path}")
    print(f"[done] roman   -> {roman_path}")
    print(f"[done] summary -> {summary_path}")
    print(f"[done] total rows processed: {total_rows}")


if __name__ == "__main__":
    main()
