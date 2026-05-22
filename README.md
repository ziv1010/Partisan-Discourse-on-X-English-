# Partisan Discourse on X - English and Multilingual Pipelines

This repository contains the code, notebooks, documentation, figures, and compact
result tables for a research workflow studying partisan discourse on X. The
project covers data cleaning, keyword extraction, manual annotation, stance model
training and evaluation, large-scale inference, multilingual translation, and
final visualization/analysis.

Large raw datasets, intermediate CSV dumps, model checkpoints, LoRA adapters,
and runtime logs are intentionally excluded from Git. The tracked repository is
meant to preserve the reproducible research code and publication-facing outputs
without pushing multi-GB artifacts to GitHub.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `codes/` | Main English pipeline, from cleaning through stance inference and visualization. |
| `Additional_Steps/` | Reviewer/revision work, multilingual translation, cross-checking, temporal splits, and final plot regeneration. |
| `Hindi_Data/` | Hindi translation/evaluation scripts used during multilingual validation. |
| `hindi_codes/` | Hindi and Hindi-English preparation, keyword extraction, stance preparation, and visualization scripts. |
| `Appendix/` | Appendix tables, comparison plots, keyword stack plots, and LaTeX support files. |
| `documentation/` and `documentation_2/` | Manuscript and supplementary LaTeX files. |
| `final_results+visualisations_folder/` | Final plotting scripts and selected compact outputs. |
| `final_visualisations/` | Publication-facing visual outputs and helper notebooks/scripts. |

## Main Workflow

1. Clean and format retweet data in `codes/1_cleaning_and_formatting/`.
2. Extract and analyze keywords in `codes/2_Keyword_Extraction/` and
   `codes/2b_keyword_analysis/`.
3. Prepare and combine manual annotations in `codes/3_manual_annotation/`.
4. Train/evaluate stance models in `codes/4_finetuning/`.
5. Extract keyword-specific tweet subsets in `codes/5_tweet_extractor/`.
6. Run stance inference in `codes/6_stance/`.
7. Generate figures and aggregate summaries in `codes/7_visualisations/`,
   `final_results+visualisations_folder/`, and `final_visualisations/`.
8. Use `Additional_Steps/` for multilingual translation, temporal checks, and
   reviewer-driven validation analyses.

## Data and Model Policy

The `.gitignore` is intentionally specific. It ignores:

- raw and large derived CSV files,
- model weights, adapters, checkpoints, and training caches,
- archives and runtime logs,
- local virtual environments and Python cache files.

Small scripts, notebooks, README files, figures, LaTeX files, compact result
tables, and text summaries should remain trackable. If a required input dataset
or model artifact is missing after cloning, restore it from local storage or the
project data archive before running the corresponding pipeline stage.

## Environment

The repository is script-oriented rather than packaged as a Python module. A
typical setup is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn matplotlib seaborn jupyter openpyxl
```

Model training and translation stages need additional GPU/ML packages depending
on the script, including combinations of:

```bash
pip install torch transformers datasets accelerate peft sentencepiece keybert
```

Some multilingual translation scripts use a separate environment documented in
`Additional_Steps/Translating/README_multilingual_translation.md`.

## Notes for Reuse

- Many scripts assume paths relative to either the repository root or their own
  folder. Check path constants before running a stage in a new environment.
- Large data files are excluded from GitHub by design; the code and compact
  audit outputs remain in version control.
- Generated outputs can be regenerated stage by stage once the excluded data and
  model artifacts are restored locally.
