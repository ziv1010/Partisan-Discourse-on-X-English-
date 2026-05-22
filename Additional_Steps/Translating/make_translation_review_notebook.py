#!/usr/bin/env python3
import json
from pathlib import Path
from textwrap import dedent


BASE_DIR = Path("/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Additional_Steps/Translating")
NOTEBOOK_PATH = BASE_DIR / "translation_pipeline_review.ipynb"
DEFAULT_RUN_DIR = BASE_DIR / "Data" / "full_multilingual_en_20260403_detached"


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip("\n").splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip("\n").splitlines(keepends=True),
    }


cells = [
    md_cell(
        f"""
        # Translation Pipeline Review

        This notebook is for auditing the multilingual translation pipeline outputs.

        Default run directory:

        - `{DEFAULT_RUN_DIR}`

        What this notebook helps with:

        - inspect branch sizes and file paths
        - sample English, native-script, and Roman/Hinglish routed rows
        - inspect translated outputs
        - inspect low-cosine examples
        - look up a specific `row_id` across stages
        - search tweets by text
        - inspect the final merged English output when it exists
        """
    ),
    code_cell(
        f"""
        from pathlib import Path
        import re
        import pandas as pd
        import numpy as np
        from IPython.display import display, Markdown, HTML

        BASE_DIR = Path("{BASE_DIR}")
        RUN_DIR = Path("{DEFAULT_RUN_DIR}")
        ROUTED_DIR = RUN_DIR / "routed"
        TRANSLATED_DIR = RUN_DIR / "translated"
        LOG_DIR = RUN_DIR / "logs"

        FILES = {{
            "english_routed": ROUTED_DIR / "english_tweets.csv",
            "native_routed": ROUTED_DIR / "hindi_script_tweets.csv",
            "roman_routed": ROUTED_DIR / "hinglish_roman_tweets.csv",
            "native_translated": TRANSLATED_DIR / "hindi_script_en.csv",
            "roman_translated": TRANSLATED_DIR / "hinglish_roman_en.csv",
            "native_lowcos": TRANSLATED_DIR / "hindi_script_lowcos.csv",
            "roman_lowcos": TRANSLATED_DIR / "hinglish_roman_lowcos.csv",
            "merged": RUN_DIR / "retweet_data_combined_labeled_by_retweet_en.csv",
            "route_log": LOG_DIR / "01_route.log",
            "native_log": LOG_DIR / "02_translate_native.log",
            "roman_log": LOG_DIR / "03_translate_roman.log",
            "merge_log": LOG_DIR / "04_merge.log",
        }}

        pd.set_option("display.max_colwidth", None)
        pd.set_option("display.max_rows", 50)

        display(
            HTML(
                '''
                <style>
                table.dataframe td, table.dataframe th {{
                    white-space: pre-wrap !important;
                    max-width: none !important;
                    overflow: visible !important;
                    text-overflow: clip !important;
                }}
                </style>
                '''
            )
        )

        summary_rows = []
        for key, path in FILES.items():
            summary_rows.append(
                {{
                    "name": key,
                    "path": str(path),
                    "exists": path.exists(),
                    "size_mb": round(path.stat().st_size / (1024 * 1024), 2) if path.exists() else None,
                }}
            )
        display(pd.DataFrame(summary_rows))
        """
    ),
    code_cell(
        """
        def route_totals_from_log(route_log: Path) -> dict:
            text = route_log.read_text(errors="ignore")
            pat = re.compile(r"\\[chunk \\d+\\] rows=\\d+ total=\\d+ english=(\\d+) native=(\\d+) roman=(\\d+)")
            english = native = roman = 0
            for e, n, r in pat.findall(text):
                english += int(e)
                native += int(n)
                roman += int(r)
            return {
                "english": english,
                "native": native,
                "roman": roman,
                "total": english + native + roman,
            }


        def last_translated_total(log_path: Path) -> int:
            if not log_path.exists():
                return 0
            text = log_path.read_text(errors="ignore")
            matches = re.findall(r"chunk \\d+ processed \\d+ rows; total (\\d+)", text)
            return int(matches[-1]) if matches else 0


        route_totals = route_totals_from_log(FILES["route_log"])
        native_done = last_translated_total(FILES["native_log"])
        roman_done = last_translated_total(FILES["roman_log"])

        display(
            pd.DataFrame(
                [
                    {"stage": "route_total", "rows": route_totals["total"]},
                    {"stage": "english_routed", "rows": route_totals["english"]},
                    {"stage": "native_routed", "rows": route_totals["native"]},
                    {"stage": "roman_routed", "rows": route_totals["roman"]},
                    {"stage": "native_translated_so_far", "rows": native_done},
                    {"stage": "roman_translated_so_far", "rows": roman_done},
                ]
            )
        )
        """
    ),
    code_cell(
        """
        def sample_csv(path: Path, n: int = 10, usecols=None, chunksize: int = 100_000, random_state: int = 42) -> pd.DataFrame:
            if not path.exists():
                return pd.DataFrame()
            rng = np.random.default_rng(random_state)
            samples = []
            for chunk in pd.read_csv(path, chunksize=chunksize, usecols=usecols):
                take = min(max(n, 1), len(chunk))
                samples.append(chunk.sample(n=take, random_state=int(rng.integers(0, 1_000_000_000))))
            if not samples:
                return pd.DataFrame(columns=usecols)
            out = pd.concat(samples, ignore_index=True)
            if len(out) > n:
                out = out.sample(n=n, random_state=random_state)
            return out.reset_index(drop=True)


        def head_csv(path: Path, n: int = 5, usecols=None) -> pd.DataFrame:
            if not path.exists():
                return pd.DataFrame()
            return pd.read_csv(path, nrows=n, usecols=usecols)


        def lookup_row(row_id: int, files=None) -> dict:
            files = files or {
                "english_routed": FILES["english_routed"],
                "native_routed": FILES["native_routed"],
                "roman_routed": FILES["roman_routed"],
                "native_translated": FILES["native_translated"],
                "roman_translated": FILES["roman_translated"],
                "merged": FILES["merged"],
            }
            found = {}
            for name, path in files.items():
                if not path.exists():
                    continue
                matches = []
                for chunk in pd.read_csv(path, chunksize=100_000):
                    hit = chunk[chunk["row_id"] == row_id]
                    if not hit.empty:
                        matches.append(hit)
                if matches:
                    found[name] = pd.concat(matches, ignore_index=True)
            return found


        def search_text(path: Path, pattern: str, text_col: str = "tweet_original", n: int = 20, case: bool = False) -> pd.DataFrame:
            if not path.exists():
                return pd.DataFrame()
            matches = []
            flags = 0 if case else re.IGNORECASE
            for chunk in pd.read_csv(path, chunksize=100_000):
                series = chunk[text_col].fillna("").astype(str)
                hit = chunk[series.str.contains(pattern, regex=True, flags=flags)]
                if not hit.empty:
                    matches.append(hit)
                if sum(len(df) for df in matches) >= n:
                    break
            if not matches:
                return pd.DataFrame()
            out = pd.concat(matches, ignore_index=True)
            return out.head(n)
        """
    ),
    md_cell(
        """
        ## Routed Branch Samples

        These cells let you inspect what the router produced before translation.
        """
    ),
    code_cell(
        """
        route_cols = [
            "row_id",
            "translation_route",
            "lid_lang",
            "lid_score",
            "tweet_original",
            "tweet",
            "nllb_src_lang",
            "xlit_lang",
        ]

        display(Markdown("### English branch sample"))
        display(sample_csv(FILES["english_routed"], n=8, usecols=route_cols))

        display(Markdown("### Native-script branch sample"))
        display(sample_csv(FILES["native_routed"], n=8, usecols=route_cols))

        display(Markdown("### Roman/Hinglish branch sample"))
        display(sample_csv(FILES["roman_routed"], n=8, usecols=route_cols))
        """
    ),
    md_cell(
        """
        ## Translated Output Samples

        These samples compare original text, normalized routed text, and final English translation.
        """
    ),
    code_cell(
        """
        translated_cols = [
            "row_id",
            "lid_lang",
            "tweet_original",
            "tweet",
            "en_text",
            "cosine_before_after",
            "translation_engine",
        ]

        display(Markdown("### Native translated sample"))
        display(sample_csv(FILES["native_translated"], n=8, usecols=translated_cols))

        display(Markdown("### Roman/Hinglish translated sample"))
        display(sample_csv(FILES["roman_translated"], n=8, usecols=translated_cols))
        """
    ),
    md_cell(
        """
        ## Low-Cosine Inspection

        These files are useful for manual quality checks. Lower cosine scores are more likely to contain weak or distorted translations.
        """
    ),
    code_cell(
        """
        def show_lowcos(path: Path, n: int = 10) -> pd.DataFrame:
            if not path.exists():
                return pd.DataFrame()
            df = pd.read_csv(path)
            if "cosine_before_after" in df.columns:
                return df.sort_values("cosine_before_after", ascending=True).head(n)
            return df.head(n)


        display(Markdown("### Native low-cosine examples"))
        display(show_lowcos(FILES["native_lowcos"], n=10))

        display(Markdown("### Roman/Hinglish low-cosine examples"))
        display(show_lowcos(FILES["roman_lowcos"], n=10))
        """
    ),
    md_cell(
        """
        ## Row Lookup

        Use `lookup_row(row_id)` to inspect one row across every available stage.
        """
    ),
    code_cell(
        """
        row_id_to_check = 166
        row_hits = lookup_row(row_id_to_check)
        for name, df in row_hits.items():
            display(Markdown(f"### {name}"))
            display(df)
        """
    ),
    md_cell(
        """
        ## Text Search

        Use this to find tweets containing a specific term or phrase in the original text.
        """
    ),
    code_cell(
        """
        search_pattern = "Oscar|Farmer|मोदी"
        display(search_text(FILES["native_routed"], search_pattern, text_col="tweet_original", n=10))
        """
    ),
    md_cell(
        """
        ## Final Merged Output

        This will only display rows once the final merge file exists.
        """
    ),
    code_cell(
        """
        merged_cols = [
            "row_id",
            "translation_route",
            "tweet_original",
            "en_text",
            "translation_engine",
        ]

        if FILES["merged"].exists():
            display(head_csv(FILES["merged"], n=10, usecols=merged_cols))
        else:
            display(Markdown("Merged file does not exist yet. Rerun this cell after the merge stage finishes."))
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2) + "\n")
print(NOTEBOOK_PATH)
