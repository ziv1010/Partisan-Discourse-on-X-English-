#!/usr/bin/env python3
import json
from pathlib import Path
from textwrap import dedent


BASE_DIR = Path("/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Additional_Steps/Translating")
NOTEBOOK_PATH = BASE_DIR / "translation_stage_audit.ipynb"
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
        # Translation Stage Audit

        This notebook is for understanding what happened at each stage of the multilingual translation pipeline run:

        - routing into English, native-script, and Roman/Hinglish branches
        - detected language mix within each routed branch
        - what non-Hindi languages were present
        - translated outputs and per-language samples
        - low-cosine review slices as a rough quality-control signal
        - final merged English output

        Default run directory:

        - `{DEFAULT_RUN_DIR}`
        """
    ),
    code_cell(
        f"""
        from pathlib import Path
        import json
        import re
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from IPython.display import display, Markdown, HTML

        plt.style.use("seaborn-v0_8-whitegrid")
        pd.set_option("display.max_colwidth", None)
        pd.set_option("display.max_rows", 200)
        pd.set_option("display.float_format", lambda x: f"{{x:,.4f}}")

        BASE_DIR = Path("{BASE_DIR}")
        RUN_DIR = Path("{DEFAULT_RUN_DIR}")
        ROUTED_DIR = RUN_DIR / "routed"
        TRANSLATED_DIR = RUN_DIR / "translated"
        LOG_DIR = RUN_DIR / "logs"
        WORK_DIR = RUN_DIR / "work" / "native_multigpu"

        FILES = {{
            "routing_summary": ROUTED_DIR / "routing_summary.csv",
            "english_routed": ROUTED_DIR / "english_tweets.csv",
            "native_routed": ROUTED_DIR / "hindi_script_tweets.csv",
            "roman_routed": ROUTED_DIR / "hinglish_roman_tweets.csv",
            "native_translated": TRANSLATED_DIR / "hindi_script_en.csv",
            "roman_translated": TRANSLATED_DIR / "hinglish_roman_en.csv",
            "native_lowcos": TRANSLATED_DIR / "hindi_script_lowcos.csv",
            "roman_lowcos": TRANSLATED_DIR / "hinglish_roman_lowcos.csv",
            "merged": RUN_DIR / "retweet_data_combined_labeled_by_retweet_en.csv",
            "manifest": WORK_DIR / "manifest.json",
            "route_log": LOG_DIR / "01_route.log",
            "native_log": LOG_DIR / "02_translate_native.log",
            "roman_log": LOG_DIR / "03_translate_roman.log",
            "assemble_log": LOG_DIR / "06_assemble_native_multigpu.log",
            "merge_log": LOG_DIR / "07_merge_after_multigpu.log",
        }}

        display(
            HTML(
                '''
                <style>
                table.dataframe td, table.dataframe th {{
                    white-space: pre-wrap !important;
                    max-width: none !important;
                    overflow: visible !important;
                    text-overflow: clip !important;
                    vertical-align: top !important;
                }}
                </style>
                '''
            )
        )
        """
    ),
    code_cell(
        """
        ROW_COUNT_CACHE = {}


        def count_csv_rows(path: Path, chunksize: int = 200_000) -> int | None:
            if not path.exists():
                return None
            key = str(path.resolve())
            if key in ROW_COUNT_CACHE:
                return ROW_COUNT_CACHE[key]
            total = 0
            for chunk in pd.read_csv(path, usecols=["row_id"], chunksize=chunksize):
                total += len(chunk)
            ROW_COUNT_CACHE[key] = total
            return total


        def file_inventory() -> pd.DataFrame:
            rows = []
            for name, path in FILES.items():
                rows.append(
                    {
                        "name": name,
                        "path": str(path),
                        "exists": path.exists(),
                        "size_mb": round(path.stat().st_size / (1024 * 1024), 2) if path.exists() else None,
                    }
                )
            return pd.DataFrame(rows).sort_values(["exists", "name"], ascending=[False, True]).reset_index(drop=True)


        def load_routing_summary() -> pd.DataFrame:
            df = pd.read_csv(FILES["routing_summary"])
            df["count"] = df["count"].astype(int)
            return df


        def route_counts_table() -> pd.DataFrame:
            df = load_routing_summary()
            out = df[df["level"] == "route"][["route", "count"]].copy()
            total = out["count"].sum()
            out["pct_total"] = out["count"] / total
            return out.sort_values("count", ascending=False).reset_index(drop=True)


        def language_counts(route: str) -> pd.DataFrame:
            df = load_routing_summary()
            route_total = int(df[(df["level"] == "route") & (df["route"] == route)]["count"].iloc[0])
            out = df[(df["level"] == "route_lang") & (df["route"] == route)][["lid_lang", "count"]].copy()
            out["pct_branch"] = out["count"] / route_total
            out["route"] = route
            return out.sort_values("count", ascending=False).reset_index(drop=True)


        def count_values(path: Path, col: str, chunksize: int = 200_000) -> pd.Series:
            counts = None
            for chunk in pd.read_csv(path, usecols=[col], chunksize=chunksize):
                vc = chunk[col].fillna("NA").astype(str).value_counts()
                counts = vc if counts is None else counts.add(vc, fill_value=0)
            if counts is None:
                return pd.Series(dtype="int64")
            return counts.sort_values(ascending=False).astype(int)


        def sample_rows(path: Path, mask_fn, columns=None, n: int = 5, chunksize: int = 100_000, random_state: int = 42) -> pd.DataFrame:
            if not path.exists():
                return pd.DataFrame(columns=columns)
            rng = np.random.default_rng(random_state)
            keep = []
            for chunk in pd.read_csv(path, chunksize=chunksize, usecols=columns):
                hit = chunk[mask_fn(chunk)]
                if hit.empty:
                    continue
                take = min(len(hit), max(1, n))
                keep.append(hit.sample(n=take, random_state=int(rng.integers(0, 1_000_000_000))))
            if not keep:
                return pd.DataFrame(columns=columns)
            out = pd.concat(keep, ignore_index=True)
            if len(out) > n:
                out = out.sample(n=n, random_state=random_state)
            return out.reset_index(drop=True)


        def sample_language(path: Path, lid_lang: str, columns=None, n: int = 3, chunksize: int = 100_000) -> pd.DataFrame:
            return sample_rows(path, lambda df: df["lid_lang"].fillna("") == lid_lang, columns=columns, n=n, chunksize=chunksize)


        def sample_top_languages(path: Path, langs: list[str], columns=None, n_per_lang: int = 2) -> pd.DataFrame:
            frames = []
            for idx, lang in enumerate(langs):
                df = sample_language(path, lang, columns=columns, n=n_per_lang, chunksize=100_000)
                if not df.empty:
                    frames.append(df)
            if not frames:
                return pd.DataFrame(columns=columns)
            return pd.concat(frames, ignore_index=True)


        def cosine_language_stats(path: Path, min_count: int = 500, chunksize: int = 200_000) -> pd.DataFrame:
            sums = {}
            counts = {}
            for chunk in pd.read_csv(path, usecols=["lid_lang", "cosine_before_after"], chunksize=chunksize):
                grp = chunk.groupby("lid_lang")["cosine_before_after"].agg(["sum", "count"])
                for lang, row in grp.iterrows():
                    sums[lang] = sums.get(lang, 0.0) + float(row["sum"])
                    counts[lang] = counts.get(lang, 0) + int(row["count"])
            rows = []
            for lang, count in counts.items():
                if count < min_count:
                    continue
                rows.append(
                    {
                        "lid_lang": lang,
                        "count": count,
                        "avg_cosine": sums[lang] / count,
                    }
                )
            return pd.DataFrame(rows).sort_values(["count", "avg_cosine"], ascending=[False, False]).reset_index(drop=True)


        def lowcos_rate_table(route: str, lowcos_path: Path) -> pd.DataFrame:
            totals = language_counts(route)[["lid_lang", "count"]].rename(columns={"count": "total_rows"})
            low = count_values(lowcos_path, "lid_lang").rename("lowcos_rows").reset_index().rename(columns={"index": "lid_lang"})
            out = totals.merge(low, on="lid_lang", how="left")
            out["lowcos_rows"] = out["lowcos_rows"].fillna(0).astype(int)
            out["lowcos_rate"] = out["lowcos_rows"] / out["total_rows"]
            return out.sort_values(["lowcos_rate", "total_rows"], ascending=[False, False]).reset_index(drop=True)


        def lookup_row(row_id: int) -> dict:
            targets = {
                "english_routed": FILES["english_routed"],
                "native_routed": FILES["native_routed"],
                "roman_routed": FILES["roman_routed"],
                "native_translated": FILES["native_translated"],
                "roman_translated": FILES["roman_translated"],
                "merged": FILES["merged"],
            }
            found = {}
            for name, path in targets.items():
                if not path.exists():
                    continue
                hits = []
                for chunk in pd.read_csv(path, chunksize=100_000):
                    match = chunk[chunk["row_id"] == row_id]
                    if not match.empty:
                        hits.append(match)
                if hits:
                    found[name] = pd.concat(hits, ignore_index=True)
            return found
        """
    ),
    md_cell(
        """
        ## File Inventory

        These are the main stage outputs, logs, and run artifacts available inside the run directory.
        """
    ),
    code_cell(
        """
        display(file_inventory())
        """
    ),
    md_cell(
        """
        ## Stage Summary

        This section summarizes the route split, translated output sizes, low-cosine review slices, and the 4-GPU native resume details.
        """
    ),
    code_cell(
        """
        route_counts = route_counts_table()
        native_rows = count_csv_rows(FILES["native_translated"])
        roman_rows = count_csv_rows(FILES["roman_translated"])
        native_lowcos_rows = count_csv_rows(FILES["native_lowcos"])
        roman_lowcos_rows = count_csv_rows(FILES["roman_lowcos"])
        merged_rows = count_csv_rows(FILES["merged"])

        stages = pd.DataFrame(
            [
                {"stage": "english_routed", "rows": int(route_counts.loc[route_counts["route"] == "english", "count"].iloc[0])},
                {"stage": "native_routed", "rows": int(route_counts.loc[route_counts["route"] == "hindi_script", "count"].iloc[0])},
                {"stage": "roman_routed", "rows": int(route_counts.loc[route_counts["route"] == "hinglish_roman", "count"].iloc[0])},
                {"stage": "native_translated", "rows": native_rows},
                {"stage": "roman_translated", "rows": roman_rows},
                {"stage": "native_lowcos_review", "rows": native_lowcos_rows},
                {"stage": "roman_lowcos_review", "rows": roman_lowcos_rows},
                {"stage": "merged_final", "rows": merged_rows},
            ]
        )
        total_rows = int(route_counts["count"].sum())
        stages["pct_of_total"] = stages["rows"] / total_rows
        display(stages)

        manifest = json.loads(FILES["manifest"].read_text()) if FILES["manifest"].exists() else None
        if manifest:
            shard_df = pd.DataFrame(manifest["shard_inputs"])
            shard_df["path"] = shard_df["path"].astype(str)
            shard_df["shard_name"] = shard_df["path"].str.extract(r"(part_\\d+\\.csv)")
            shard_df = shard_df[["shard_name", "rows", "path"]]
            display(
                pd.DataFrame(
                    [
                        {"metric": "native_total_rows", "value": manifest["native_total_rows"]},
                        {"metric": "prefix_rows_completed_before_4gpu_resume", "value": manifest["prefix_rows"]},
                        {"metric": "remaining_rows_sent_to_4gpu_resume", "value": manifest["remaining_rows"]},
                        {"metric": "num_shards", "value": len(manifest["shard_inputs"])},
                    ]
                )
            )
            display(shard_df)
        """
    ),
    code_cell(
        """
        native_langs = language_counts("hindi_script")
        roman_langs = language_counts("hinglish_roman")
        english_langs = language_counts("english")

        fig, axes = plt.subplots(1, 3, figsize=(20, 5))

        axes[0].bar(route_counts["route"], route_counts["count"], color=["#2b8cbe", "#de2d26", "#31a354"])
        axes[0].set_title("Route Split")
        axes[0].set_ylabel("Rows")
        axes[0].tick_params(axis="x", rotation=20)

        native_top = native_langs.head(12).sort_values("count")
        axes[1].barh(native_top["lid_lang"], native_top["count"], color="#de2d26")
        axes[1].set_title("Top Native-Script Languages")
        axes[1].set_xlabel("Rows")

        roman_top = roman_langs.head(12).sort_values("count")
        axes[2].barh(roman_top["lid_lang"], roman_top["count"], color="#31a354")
        axes[2].set_title("Top Roman-Script Detections")
        axes[2].set_xlabel("Rows")

        plt.tight_layout()
        plt.show()
        """
    ),
    md_cell(
        """
        ## Are There Languages Other Than Hindi?

        Yes. This section quantifies them directly from the routing summary.
        """
    ),
    code_cell(
        """
        native_hindi = int(native_langs.loc[native_langs["lid_lang"] == "hin_Deva", "count"].iloc[0])
        native_other = int(native_langs.loc[native_langs["lid_lang"] != "hin_Deva", "count"].sum())
        roman_hindi = int(roman_langs.loc[roman_langs["lid_lang"] == "hin_Latn", "count"].iloc[0])
        roman_other = int(roman_langs.loc[roman_langs["lid_lang"] != "hin_Latn", "count"].sum())
        english_other = int(english_langs.loc[english_langs["lid_lang"] == "other", "count"].iloc[0])

        non_hindi_summary = pd.DataFrame(
            [
                {"branch": "native_script", "hindi_rows": native_hindi, "non_hindi_rows": native_other, "non_hindi_pct": native_other / (native_hindi + native_other)},
                {"branch": "roman_hinglish", "hindi_rows": roman_hindi, "non_hindi_rows": roman_other, "non_hindi_pct": roman_other / (roman_hindi + roman_other)},
                {"branch": "english_passthrough", "hindi_rows": None, "non_hindi_rows": english_other, "non_hindi_pct": english_other / int(english_langs["count"].sum())},
            ]
        )
        display(non_hindi_summary)

        display(Markdown("### Top non-Hindi native-script languages"))
        display(native_langs.query("lid_lang != 'hin_Deva'").head(15))

        display(Markdown("### Top non-Hindi Roman-script detections"))
        display(roman_langs.query("lid_lang != 'hin_Latn'").head(15))
        """
    ),
    code_cell(
        """
        mix_df = pd.DataFrame(
            [
                {"branch": "native_script", "group": "Hindi", "rows": native_hindi},
                {"branch": "native_script", "group": "Other languages", "rows": native_other},
                {"branch": "roman_hinglish", "group": "Hindi", "rows": roman_hindi},
                {"branch": "roman_hinglish", "group": "Other languages", "rows": roman_other},
            ]
        )
        mix_pivot = mix_df.pivot(index="branch", columns="group", values="rows")

        ax = mix_pivot.plot(kind="barh", stacked=True, figsize=(10, 4), color=["#3182bd", "#f16913"])
        ax.set_title("Hindi vs Other Languages by Branch")
        ax.set_xlabel("Rows")
        plt.tight_layout()
        plt.show()
        """
    ),
    md_cell(
        """
        ## Sample Outputs By Language

        These samples show routed or translated rows for non-Hindi detections so you can see what those labels actually look like in the data.
        """
    ),
    code_cell(
        """
        english_other_cols = ["row_id", "lid_lang", "tweet_original", "tweet", "translation_route"]
        translated_cols = ["row_id", "lid_lang", "nllb_src_lang", "tweet_original", "tweet", "en_text", "cosine_before_after"]

        display(Markdown("### English branch rows labeled as `other`"))
        display(sample_rows(FILES["english_routed"], lambda df: df["lid_lang"].fillna("") == "other", columns=english_other_cols, n=8))

        top_native_non_hindi = native_langs.query("lid_lang != 'hin_Deva'").head(8)["lid_lang"].tolist()
        display(Markdown("### Native-script translated samples for top non-Hindi languages"))
        for lang in top_native_non_hindi:
            display(Markdown(f"#### {lang}"))
            display(sample_language(FILES["native_translated"], lang, columns=translated_cols, n=3))

        top_roman_langs = roman_langs.head(8)["lid_lang"].tolist()
        display(Markdown("### Roman/Hinglish translated samples for top detected languages"))
        for lang in top_roman_langs:
            display(Markdown(f"#### {lang}"))
            display(sample_language(FILES["roman_translated"], lang, columns=translated_cols, n=3))
        """
    ),
    md_cell(
        """
        ## Low-Cosine Review Slices

        The low-cos files are the bottom-quantile manual-review slices. These are not necessarily wrong, but they are the main candidates for inspection.
        """
    ),
    code_cell(
        """
        native_lowcos_rates = lowcos_rate_table("hindi_script", FILES["native_lowcos"])
        roman_lowcos_rates = lowcos_rate_table("hinglish_roman", FILES["roman_lowcos"])

        display(Markdown("### Native-script low-cosine rates"))
        display(native_lowcos_rates.head(15))

        display(Markdown("### Roman/Hinglish low-cosine rates"))
        display(roman_lowcos_rates.head(15))
        """
    ),
    code_cell(
        """
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        native_plot = native_lowcos_rates.query("total_rows >= 1000").head(12).sort_values("lowcos_rate")
        axes[0].barh(native_plot["lid_lang"], native_plot["lowcos_rate"], color="#de2d26")
        axes[0].set_title("Native-Script Low-Cosine Rate")
        axes[0].set_xlabel("Low-cosine rows / total rows")

        roman_plot = roman_lowcos_rates.query("total_rows >= 200").head(12).sort_values("lowcos_rate")
        axes[1].barh(roman_plot["lid_lang"], roman_plot["lowcos_rate"], color="#31a354")
        axes[1].set_title("Roman-Script Low-Cosine Rate")
        axes[1].set_xlabel("Low-cosine rows / total rows")

        plt.tight_layout()
        plt.show()
        """
    ),
    code_cell(
        """
        display(Markdown("### Native low-cosine examples"))
        display(pd.read_csv(FILES["native_lowcos"]).sort_values("cosine_before_after").head(12)[translated_cols])

        display(Markdown("### Roman/Hinglish low-cosine examples"))
        display(pd.read_csv(FILES["roman_lowcos"]).sort_values("cosine_before_after").head(12)[translated_cols])
        """
    ),
    md_cell(
        """
        ## Translation Quality By Language

        This cell scans the translated outputs and computes mean cosine by language. It is heavier than the routing-summary cells.
        """
    ),
    code_cell(
        """
        native_cos = cosine_language_stats(FILES["native_translated"], min_count=1000)
        roman_cos = cosine_language_stats(FILES["roman_translated"], min_count=200)

        display(Markdown("### Native-script average cosine by language"))
        display(native_cos.head(20))

        display(Markdown("### Roman/Hinglish average cosine by language"))
        display(roman_cos.head(20))
        """
    ),
    code_cell(
        """
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        native_cos_plot = native_cos.head(12).sort_values("avg_cosine")
        axes[0].barh(native_cos_plot["lid_lang"], native_cos_plot["avg_cosine"], color="#9e9ac8")
        axes[0].set_xlim(0, 1)
        axes[0].set_title("Native-Script Mean Cosine")
        axes[0].set_xlabel("Average cosine_before_after")

        roman_cos_plot = roman_cos.head(12).sort_values("avg_cosine")
        axes[1].barh(roman_cos_plot["lid_lang"], roman_cos_plot["avg_cosine"], color="#74c476")
        axes[1].set_xlim(0, 1)
        axes[1].set_title("Roman/Hinglish Mean Cosine")
        axes[1].set_xlabel("Average cosine_before_after")

        plt.tight_layout()
        plt.show()
        """
    ),
    md_cell(
        """
        ## Final Output And Lookup Tools

        Use these cells to inspect the merged English output or trace a specific `row_id` across stages.
        """
    ),
    code_cell(
        """
        merged_cols = ["row_id", "translation_route", "lid_lang", "tweet_original", "en_text", "translation_engine", "cosine_before_after"]

        display(Markdown("### Final merged output sample"))
        display(pd.read_csv(FILES["merged"], nrows=12, usecols=merged_cols))

        display(Markdown("### Final merged route split"))
        display(pd.read_csv(FILES["merged"], usecols=["translation_route"]).value_counts().rename("rows").reset_index())
        """
    ),
    code_cell(
        """
        row_id_to_check = 166
        row_hits = lookup_row(row_id_to_check)
        for stage_name, df in row_hits.items():
            display(Markdown(f"### {stage_name}"))
            display(df)
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
