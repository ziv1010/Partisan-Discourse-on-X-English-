#!/usr/bin/env python3
"""
Build temporal LoRA finetuning splits.

Outputs are written beside this script:
- temporal_joint_dataset_raw.csv
- temporal_joint_dataset_with_dates.csv
- temporal_master_train_2020_2021.csv
- temporal_master_test_2022_2023.csv
- temporal_rows_missing_dates.csv
- temporal_rows_missing_stance.csv
- temporal_split_summary.json

The training/test split is by tweet year:
- train: 2020 and 2021
- test : 2022 and 2023
"""

from __future__ import annotations

import html
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]

EXCEL_PATH = BASE_DIR / "Annotations_Jan'26-3.xlsx"
DATA_FORMATTING_DIR = PROJECT_ROOT / "codes" / "4_finetuning" / "4_a_DataProcessing" / "data_formatting"
MASTER_TRAIN = DATA_FORMATTING_DIR / "master_train.csv"
MASTER_TEST = DATA_FORMATTING_DIR / "master_test.csv"

POSTJOINING_FINAL = (
    PROJECT_ROOT
    / "Additional_Steps"
    / "Crosschecking_and_Joining"
    / "PostJoining_stance"
    / "extracted_ALL_35keywords_english_hindi_only_final_with_existing_and_inferred_stance.csv"
)
FINAL_STANCE = (
    PROJECT_ROOT
    / "Additional_Steps"
    / "Crosschecking_and_Joining"
    / "FINAL_STANCE"
    / "extracted_ALL_35keywords_english_hindi_only_final_with_existing_and_inferred_stance.csv"
)
BRIDGE_CSV = (
    PROJECT_ROOT
    / "Additional_Steps"
    / "Crosschecking_and_Joining"
    / "SCRIPTS_OuTPUTS"
    / "retweet_data_combined_labeled_by_retweet_en_with_hinglish_bridge.csv"
)

RAW_JOINT_OUT = BASE_DIR / "temporal_joint_dataset_raw.csv"
DATED_JOINT_OUT = BASE_DIR / "temporal_joint_dataset_with_dates.csv"
TRAIN_OUT = BASE_DIR / "temporal_master_train_2020_2021.csv"
TEST_OUT = BASE_DIR / "temporal_master_test_2022_2023.csv"
MISSING_DATES_OUT = BASE_DIR / "temporal_rows_missing_dates.csv"
MISSING_STANCE_OUT = BASE_DIR / "temporal_rows_missing_stance.csv"
SUMMARY_OUT = BASE_DIR / "temporal_split_summary.json"

VALID_STANCES = {"For", "Against", "Neutral"}
TRAIN_YEARS = {2020, 2021}
TEST_YEARS = {2022, 2023}

BASE_COLUMNS = [
    "joint_row_id",
    "row_origin",
    "source_file",
    "sheet_name",
    "source_row",
    "tweet",
    "tweet_label",
    "_label_norm",
    "keyword",
    "subjects",
    "subjects_scored",
    "stance",
    "stance_raw",
    "stance_reason",
    "stance_annotation_column",
    "fewshot_label",
    "fewshot_reason",
    "fewshot_raw",
    "fewshot_shots_json",
    "original_author",
    "excel_secondary_stance",
    "excel_secondary_stance_column",
]

INTERNAL_MATCH_COLUMNS = [
    "keyword_norm",
    "tweet_norm",
    "source_row_norm",
    "match_key",
    "source_keyword_key",
]


@dataclass
class FinalMatch:
    row_id: str
    source_row: str
    keyword: str
    tweet: str
    language: str
    text_column: str


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text.strip()


def normalize_header(value: Any) -> str:
    text = clean_cell(value).lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def normalize_keyword(value: Any) -> str:
    text = unicodedata.normalize("NFKC", html.unescape(clean_cell(value))).lower()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_tweet(value: Any) -> str:
    text = unicodedata.normalize("NFKC", html.unescape(clean_cell(value)))
    text = text.replace("\u2066", "").replace("\u2069", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_id(value: Any) -> str:
    text = clean_cell(value)
    if not text:
        return ""
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def normalize_stance(value: Any) -> str:
    text = unicodedata.normalize("NFKC", clean_cell(value)).lower()
    key = re.sub(r"[^a-z]+", "", text)
    if key in {"for", "favor", "favour", "favors", "favours", "support", "supports", "positive", "pro"}:
        return "For"
    if key in {"against", "againts", "anti", "oppose", "opposes", "negative", "deny", "denies"}:
        return "Against"
    if key in {"neutral", "nuetral", "neutal", "none"}:
        return "Neutral"
    return ""


def first_existing_column(columns: list[str], candidates: list[str]) -> str:
    normalized = {normalize_header(c): c for c in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return ""


def find_indrabati_stance_column(columns: list[str]) -> str:
    for column in columns:
        key = normalize_header(column)
        if "indrabati" in key and "stance" in key:
            return column
    return ""


def row_value(row: pd.Series, column: str) -> str:
    if not column or column not in row.index:
        return ""
    return clean_cell(row[column])


def read_master(path: Path, origin: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype="string", keep_default_na=False)
    records: list[dict[str, str]] = []
    for _, row in df.iterrows():
        stance_raw = row_value(row, "stance")
        records.append(
            {
                "row_origin": origin,
                "source_file": str(path),
                "sheet_name": "",
                "source_row": normalize_id(row_value(row, "source_row")),
                "tweet": row_value(row, "tweet"),
                "tweet_label": row_value(row, "tweet_label"),
                "_label_norm": "",
                "keyword": row_value(row, "keyword"),
                "subjects": row_value(row, "subjects"),
                "subjects_scored": row_value(row, "subjects_scored"),
                "stance": normalize_stance(stance_raw),
                "stance_raw": stance_raw,
                "stance_reason": row_value(row, "stance_reason"),
                "stance_annotation_column": "stance",
                "fewshot_label": "",
                "fewshot_reason": "",
                "fewshot_raw": "",
                "fewshot_shots_json": "",
                "original_author": "",
                "excel_secondary_stance": "",
                "excel_secondary_stance_column": "",
            }
        )
    return pd.DataFrame.from_records(records, columns=BASE_COLUMNS[1:])


def read_excel_annotations(path: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    records: list[dict[str, str]] = []

    for sheet_name in xl.sheet_names:
        if sheet_name.strip().lower() == "legend":
            continue

        df = pd.read_excel(path, sheet_name=sheet_name, dtype="string", keep_default_na=False)
        columns = list(df.columns)

        source_row_col = first_existing_column(columns, ["source_row"])
        tweet_col = first_existing_column(columns, ["tweet", "text", "statement"])
        original_author_col = first_existing_column(columns, ["original_author"])
        tweet_label_col = first_existing_column(columns, ["tweet_label"])
        label_norm_col = first_existing_column(columns, ["label_norm"])
        keyword_col = first_existing_column(columns, ["keyword", "entity", "topic"])
        subjects_scored_col = first_existing_column(columns, ["subjects_scored"])
        fewshot_raw_col = first_existing_column(columns, ["fewshot_raw"])
        fewshot_label_col = first_existing_column(columns, ["fewshot_label"])
        fewshot_reason_col = first_existing_column(columns, ["fewshot_reason"])
        fewshot_shots_col = first_existing_column(columns, ["fewshot_shots_json"])
        stance_primary_col = first_existing_column(columns, ["stance"])
        stance_gold_col = first_existing_column(columns, ["stance_gold"])
        reasoning_col = first_existing_column(columns, ["reasoning", "reason"])
        indrabat_stance_col = find_indrabati_stance_column(columns)

        for _, row in df.iterrows():
            primary_raw = row_value(row, stance_primary_col)
            secondary_raw = row_value(row, indrabat_stance_col)
            gold_raw = row_value(row, stance_gold_col)

            stance = normalize_stance(primary_raw)
            stance_col_used = stance_primary_col if stance else ""
            stance_raw = primary_raw

            if not stance:
                stance = normalize_stance(secondary_raw)
                stance_col_used = indrabat_stance_col if stance else ""
                stance_raw = secondary_raw if stance else primary_raw

            if not stance:
                stance = normalize_stance(gold_raw)
                stance_col_used = stance_gold_col if stance else ""
                stance_raw = gold_raw if stance else primary_raw

            records.append(
                {
                    "row_origin": "excel_annotations",
                    "source_file": str(path),
                    "sheet_name": sheet_name,
                    "source_row": normalize_id(row_value(row, source_row_col)),
                    "tweet": row_value(row, tweet_col),
                    "tweet_label": row_value(row, tweet_label_col),
                    "_label_norm": row_value(row, label_norm_col),
                    "keyword": row_value(row, keyword_col) or sheet_name.strip().lower(),
                    "subjects": "",
                    "subjects_scored": row_value(row, subjects_scored_col),
                    "stance": stance,
                    "stance_raw": stance_raw,
                    "stance_reason": row_value(row, reasoning_col),
                    "stance_annotation_column": stance_col_used,
                    "fewshot_label": row_value(row, fewshot_label_col),
                    "fewshot_reason": row_value(row, fewshot_reason_col),
                    "fewshot_raw": row_value(row, fewshot_raw_col),
                    "fewshot_shots_json": row_value(row, fewshot_shots_col),
                    "original_author": row_value(row, original_author_col),
                    "excel_secondary_stance": secondary_raw,
                    "excel_secondary_stance_column": indrabat_stance_col,
                }
            )

    return pd.DataFrame.from_records(records, columns=BASE_COLUMNS[1:])


def make_match_key(keyword: Any, tweet: Any) -> str:
    return f"{normalize_keyword(keyword)}\u241f{normalize_tweet(tweet)}"


def build_final_lookup(target: pd.DataFrame, final_csv: Path) -> tuple[dict[str, FinalMatch], dict[str, FinalMatch]]:
    target_text_keys = set(target["match_key"])
    target_source_keyword_keys = set(target["source_keyword_key"])
    target_source_keyword_keys.discard("\u241f")

    text_lookup: dict[str, FinalMatch] = {}
    source_lookup: dict[str, FinalMatch] = {}

    usecols = ["source_row", "row_id", "keyword", "tweet", "tweet_original", "en_text", "language"]
    for i, chunk in enumerate(pd.read_csv(final_csv, usecols=usecols, dtype="string", chunksize=250_000, low_memory=False)):
        chunk = chunk.fillna("")
        chunk["keyword_norm"] = chunk["keyword"].map(normalize_keyword)
        chunk["source_row_norm"] = chunk["source_row"].map(normalize_id)
        chunk["row_id_norm"] = chunk["row_id"].map(normalize_id)

        for id_col in ["source_row_norm", "row_id_norm"]:
            keys = chunk["keyword_norm"] + "\u241f" + chunk[id_col]
            mask = keys.isin(target_source_keyword_keys)
            if mask.any():
                for key, row in chunk.loc[mask].assign(_key=keys[mask]).iterrows():
                    lookup_key = row["_key"]
                    source_lookup.setdefault(
                        lookup_key,
                        FinalMatch(
                            row_id=normalize_id(row["row_id"]),
                            source_row=normalize_id(row["source_row"]),
                            keyword=clean_cell(row["keyword"]),
                            tweet=clean_cell(row["tweet"]),
                            language=clean_cell(row["language"]),
                            text_column=id_col,
                        ),
                    )

        for text_col in ["tweet", "en_text", "tweet_original"]:
            text_keys = chunk["keyword_norm"] + "\u241f" + chunk[text_col].map(normalize_tweet)
            mask = text_keys.isin(target_text_keys)
            if not mask.any():
                continue
            for _, row in chunk.loc[mask].assign(_key=text_keys[mask]).iterrows():
                lookup_key = row["_key"]
                text_lookup.setdefault(
                    lookup_key,
                    FinalMatch(
                        row_id=normalize_id(row["row_id"]),
                        source_row=normalize_id(row["source_row"]),
                        keyword=clean_cell(row["keyword"]),
                        tweet=clean_cell(row[text_col]),
                        language=clean_cell(row["language"]),
                        text_column=text_col,
                    ),
                )

        if (i + 1) % 5 == 0:
            print(f"  scanned {(i + 1) * 250_000:,} final stance rows")

    return text_lookup, source_lookup


def build_bridge_lookup(row_ids: set[str]) -> dict[str, dict[str, str]]:
    row_ids = {normalize_id(v) for v in row_ids if normalize_id(v)}
    lookup: dict[str, dict[str, str]] = {}
    usecols = ["row_id", "timestamp", "tweet", "tweet_original", "en_text"]

    for i, chunk in enumerate(pd.read_csv(BRIDGE_CSV, usecols=usecols, dtype="string", chunksize=250_000, low_memory=False)):
        chunk = chunk.fillna("")
        chunk["row_id_norm"] = chunk["row_id"].map(normalize_id)
        matched = chunk[chunk["row_id_norm"].isin(row_ids)]
        for _, row in matched.iterrows():
            rid = normalize_id(row["row_id"])
            lookup.setdefault(
                rid,
                {
                    "timestamp": clean_cell(row["timestamp"]),
                    "bridge_tweet": clean_cell(row["tweet"]),
                    "bridge_tweet_original": clean_cell(row["tweet_original"]),
                    "bridge_en_text": clean_cell(row["en_text"]),
                },
            )

        if (i + 1) % 10 == 0:
            print(f"  scanned {(i + 1) * 250_000:,} bridge rows")

    return lookup


def add_final_matches(joint: pd.DataFrame) -> pd.DataFrame:
    joint = joint.copy()
    joint["keyword_norm"] = joint["keyword"].map(normalize_keyword)
    joint["tweet_norm"] = joint["tweet"].map(normalize_tweet)
    joint["source_row_norm"] = joint["source_row"].map(normalize_id)
    joint["match_key"] = joint.apply(lambda r: make_match_key(r["keyword"], r["tweet"]), axis=1)
    joint["source_keyword_key"] = joint["keyword_norm"] + "\u241f" + joint["source_row_norm"]

    print("\nBuilding exact tweet/date lookup from FINAL_STANCE...")
    text_lookup, source_lookup = build_final_lookup(joint, FINAL_STANCE)
    print(f"  matched tweet+keyword keys found: {len(text_lookup):,}")
    print(f"  matched source_row/row_id+keyword keys found: {len(source_lookup):,}")

    matched_row_ids: list[str] = []
    matched_source_rows: list[str] = []
    matched_keywords: list[str] = []
    matched_languages: list[str] = []
    matched_text_columns: list[str] = []
    matched_strategies: list[str] = []

    for _, row in joint.iterrows():
        match = text_lookup.get(row["match_key"])
        strategy = "exact_tweet_keyword"
        if match is None:
            match = source_lookup.get(row["source_keyword_key"])
            strategy = "source_or_row_id_keyword"

        if match is None:
            matched_row_ids.append("")
            matched_source_rows.append("")
            matched_keywords.append("")
            matched_languages.append("")
            matched_text_columns.append("")
            matched_strategies.append("unmatched")
        else:
            matched_row_ids.append(match.row_id)
            matched_source_rows.append(match.source_row)
            matched_keywords.append(match.keyword)
            matched_languages.append(match.language)
            matched_text_columns.append(match.text_column)
            matched_strategies.append(strategy)

    joint["matched_final_row_id"] = matched_row_ids
    joint["matched_final_source_row"] = matched_source_rows
    joint["matched_final_keyword"] = matched_keywords
    joint["matched_final_language"] = matched_languages
    joint["matched_final_text_column"] = matched_text_columns
    joint["date_match_strategy"] = matched_strategies
    return joint


def add_dates(joint: pd.DataFrame) -> pd.DataFrame:
    row_ids = set(joint["matched_final_row_id"].dropna().astype(str))
    row_ids.update(joint["source_row_norm"].dropna().astype(str))

    print("\nBuilding timestamp lookup from bridge CSV...")
    bridge_lookup = build_bridge_lookup(row_ids)
    print(f"  bridge row_ids found: {len(bridge_lookup):,}")

    timestamps: list[str] = []
    bridge_strategies: list[str] = []

    for _, row in joint.iterrows():
        timestamp = ""
        bridge_strategy = ""
        rid = normalize_id(row["matched_final_row_id"])
        source_row = normalize_id(row["source_row_norm"])

        if rid and rid in bridge_lookup:
            timestamp = bridge_lookup[rid]["timestamp"]
            bridge_strategy = "matched_final_row_id"
        elif source_row and source_row in bridge_lookup:
            candidate = bridge_lookup[source_row]
            row_tweet_norm = row["tweet_norm"]
            bridge_texts = [
                normalize_tweet(candidate["bridge_tweet"]),
                normalize_tweet(candidate["bridge_tweet_original"]),
                normalize_tweet(candidate["bridge_en_text"]),
            ]
            if row_tweet_norm in bridge_texts:
                timestamp = candidate["timestamp"]
                bridge_strategy = "source_row_exact_tweet_bridge"
            else:
                timestamp = candidate["timestamp"]
                bridge_strategy = "source_row_bridge"

        timestamps.append(timestamp)
        bridge_strategies.append(bridge_strategy or "unmatched")

    joint["timestamp"] = timestamps
    parsed = pd.to_datetime(joint["timestamp"], errors="coerce", utc=True)
    joint["date"] = parsed.dt.date.astype("string").fillna("")
    joint["year"] = parsed.dt.year.astype("Int64")
    joint["date_bridge_strategy"] = bridge_strategies
    return joint


def make_temporal_splits(joint: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid = joint[joint["stance"].isin(VALID_STANCES)].copy()
    missing_stance = joint[~joint["stance"].isin(VALID_STANCES)].copy()
    missing_dates = valid[valid["year"].isna()].copy()
    valid = valid[valid["year"].notna()].copy()
    valid["year_int"] = valid["year"].astype(int)

    train = valid[valid["year_int"].isin(TRAIN_YEARS)].copy()
    test = valid[valid["year_int"].isin(TEST_YEARS)].copy()

    sort_cols = ["year_int", "date", "keyword", "joint_row_id"]
    train = train.sort_values(sort_cols).drop(columns=["year_int"])
    test = test.sort_values(sort_cols).drop(columns=["year_int"])
    return train, test, missing_dates, missing_stance


def drop_internal_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in INTERNAL_MATCH_COLUMNS if c in df.columns])


def summarize(joint: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, missing_dates: pd.DataFrame, missing_stance: pd.DataFrame) -> dict[str, Any]:
    def value_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
        return {str(k): int(v) for k, v in df[column].fillna("").value_counts(dropna=False).sort_index().items()}

    def grouped_counts(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
        if df.empty:
            return []
        out = df.groupby(columns, dropna=False).size().reset_index(name="rows")
        return json.loads(out.to_json(orient="records"))

    return {
        "inputs": {
            "excel": str(EXCEL_PATH),
            "master_train": str(MASTER_TRAIN),
            "master_test": str(MASTER_TEST),
            "postjoining_final_checked": str(POSTJOINING_FINAL),
            "final_stance_used_for_matching": str(FINAL_STANCE),
            "bridge_used_for_dates": str(BRIDGE_CSV),
        },
        "outputs": {
            "raw_joint": str(RAW_JOINT_OUT),
            "dated_joint": str(DATED_JOINT_OUT),
            "train_2020_2021": str(TRAIN_OUT),
            "test_2022_2023": str(TEST_OUT),
            "missing_dates": str(MISSING_DATES_OUT),
            "missing_stance": str(MISSING_STANCE_OUT),
        },
        "rows": {
            "joint_total": int(len(joint)),
            "train_2020_2021": int(len(train)),
            "test_2022_2023": int(len(test)),
            "missing_dates_among_valid_stance": int(len(missing_dates)),
            "missing_stance": int(len(missing_stance)),
        },
        "joint_by_origin": value_counts(joint, "row_origin"),
        "joint_stance_counts": value_counts(joint, "stance"),
        "date_match_strategy": value_counts(joint, "date_match_strategy"),
        "date_bridge_strategy": value_counts(joint, "date_bridge_strategy"),
        "joint_year_counts": value_counts(joint[joint["year"].notna()].astype({"year": "string"}), "year"),
        "train_by_year_stance": grouped_counts(train, ["year", "stance"]),
        "test_by_year_stance": grouped_counts(test, ["year", "stance"]),
        "train_by_origin": value_counts(train, "row_origin"),
        "test_by_origin": value_counts(test, "row_origin"),
    }


def main() -> None:
    print("Reading old master train/test CSVs...")
    master_train = read_master(MASTER_TRAIN, "master_train_original")
    master_test = read_master(MASTER_TEST, "master_test_original")

    print("Reading Excel annotation workbook...")
    excel = read_excel_annotations(EXCEL_PATH)

    joint = pd.concat([master_train, master_test, excel], ignore_index=True)
    joint.insert(0, "joint_row_id", range(len(joint)))
    joint["tweet_keyword_duplicate_count"] = joint.groupby(
        [joint["keyword"].map(normalize_keyword), joint["tweet"].map(normalize_tweet)], dropna=False
    )["joint_row_id"].transform("size")

    print(f"Writing raw joint dataset: {RAW_JOINT_OUT}")
    joint.to_csv(RAW_JOINT_OUT, index=False)

    joint = add_final_matches(joint)
    joint = add_dates(joint)
    joint = drop_internal_columns(joint)

    print(f"Writing dated joint dataset: {DATED_JOINT_OUT}")
    joint.to_csv(DATED_JOINT_OUT, index=False)

    train, test, missing_dates, missing_stance = make_temporal_splits(joint)

    print(f"Writing temporal train split: {TRAIN_OUT}")
    train.to_csv(TRAIN_OUT, index=False)
    print(f"Writing temporal test split: {TEST_OUT}")
    test.to_csv(TEST_OUT, index=False)
    print(f"Writing missing-date audit file: {MISSING_DATES_OUT}")
    missing_dates.to_csv(MISSING_DATES_OUT, index=False)
    print(f"Writing missing-stance audit file: {MISSING_STANCE_OUT}")
    missing_stance.to_csv(MISSING_STANCE_OUT, index=False)

    summary = summarize(joint, train, test, missing_dates, missing_stance)
    print(f"Writing summary: {SUMMARY_OUT}")
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nDone.")
    print(json.dumps(summary["rows"], indent=2))


if __name__ == "__main__":
    main()
