# Main Code Pipeline

The `codes/` directory contains the primary English-language research pipeline.
It is organized in the order the analysis was developed and run.

## Stages

| Directory | Contents |
| --- | --- |
| `1_cleaning_and_formatting/` | Data cleaning, label preparation, and polarity checks. |
| `2_Keyword_Extraction/` | Subject/keyword extraction scripts and formatter notebooks. |
| `2b_keyword_analysis/` | Keyword frequency, stance distribution, and keyword selection analyses. |
| `3_manual_annotation/` | Annotation notebooks and utilities for combining/deduplicating sheets. |
| `4_finetuning/` | Stance model preprocessing, training, evaluation, McNemar tests, multi-seed runs, and temporal split experiments. |
| `5_tweet_extractor/` | Keyword-based tweet extraction utilities and extracted tweet ID lists. |
| `6_stance/` | Stance fine-tuning and parallel inference scripts. |
| `7_visualisations/` | Plotting notebooks and scripts for stance, polarization, keyword, and party-level visualizations. |
| `8_manual_analysis/` | Utilities for packaging stance outputs for manual review. |

## Running Code

Most scripts can be run from their containing directory after restoring the
required input files. The largest raw datasets, model checkpoints, and generated
inference outputs are ignored by Git, so a fresh clone will need those restored
locally before end-to-end execution.

For reproducible experiments, prefer the wrapper scripts in `4_finetuning/` and
the documented subfolder README files where present.
