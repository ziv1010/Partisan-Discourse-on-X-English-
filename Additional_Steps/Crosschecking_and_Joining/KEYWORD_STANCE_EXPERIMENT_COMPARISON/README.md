# Keyword Stance Experiment Comparison

This folder contains keyword-specific stance tables for the normal stance experiment and the keyword temporal stance experiment.

## Main comparison

The main comparison uses `TEMPORAL_SPLIT/split_2022_2023_lora_labeled.csv` because it contains both:

- normal stance in `fewshot_label_for_against`
- keyword temporal stance in `lora_stance`

That makes `joint_normal_vs_temporal_2022_2023_table.tex` and `joint_experiment_comparison_2022_2023.csv` apples-to-apples comparisons on the exact same 2022-2023 tweet-keyword rows.

## Generated files

- `joint_normal_vs_temporal_2022_2023_table.tex`: side-by-side LaTeX table for both experiments.
- `joint_experiment_comparison_2022_2023.csv`: wide CSV with normal %, temporal %, and temporal-minus-normal percentage-point deltas.
- `full_normal_vs_temporal_percentage_comparison.csv`: non-matched percentage comparison using all normal labels and the temporal-label subset.
- `normal_full_keyword_stance_table.tex`: reference LaTeX table for the full `FINAL_STANCE` normal labels.
- `normal_matched_2022_2023_keyword_stance_table.tex`: normal-label LaTeX table on the shared temporal split rows.
- `temporal_lora_2022_2023_keyword_stance_table.tex`: temporal-label LaTeX table on the shared temporal split rows.
- `all_keyword_stance_tables.tex`: all LaTeX table snippets in one file.
- `stance_distribution_long.csv`: tidy long-format CSV for all generated experiment scopes.

## LaTeX packages

Use these packages in your paper preamble:

```tex
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{adjustbox}
```

## Run

```bash
python3 build_keyword_stance_comparison.py
```

## Row accounting

- normal_full_2020_2023: 545,250 usable rows from 545,250 read rows; label column `fewshot_label_for_against`.
- normal_matched_2022_2023: 239,826 usable rows from 239,826 read rows; label column `fewshot_label_for_against`.
- keyword_temporal_lora_2022_2023: 239,826 usable rows from 239,826 read rows; label column `lora_stance`.
