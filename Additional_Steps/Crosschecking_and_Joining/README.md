# Cross-Checking, Joining, and Final Stance Outputs

This directory contains scripts used to connect the English and multilingual
pipelines, audit coverage, backfill existing stance labels, run temporal splits,
and regenerate final visualizations requested during review.

## Key Areas

| Directory | Contents |
| --- | --- |
| `SCRIPTS_OuTPUTS/` | Coverage checks, missing-reason classification, keyword extraction updates, and merge/backfill scripts. |
| `extracted_by_keyword/` | Keyword extraction outputs and ID lists used by joining checks. Large CSV extracts are ignored. |
| `PostJoining_stance/` | Post-join inference outputs and summaries. Large inference CSVs are ignored. |
| `TEMPORAL_SPLIT/` | Temporal train/test split scripts and temporal LoRA inference utilities. |
| `FINAL_STANCE/` | Final combined visualizations and scripts for publication-facing stance plots. |
| `KEYWORD_STANCE_EXPERIMENT_COMPARISON/` | Comparison tables for normal and temporal keyword stance experiments. |

Scripts and compact text/figure outputs are intended to be committed. Raw joins,
full extracted tweet CSVs, and parallel inference state folders are kept local.
