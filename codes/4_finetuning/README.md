# Stance Model Training and Evaluation

This folder contains the stance-classification model workflow: preprocessing,
baseline/fine-tuned model training, evaluation, multi-seed comparison, temporal
split experiments, and McNemar significance tests.

## Main Areas

| Directory/File | Purpose |
| --- | --- |
| `4_a_DataProcessing/` | Data preparation for fine-tuning. |
| `4_b_TestingComparison/` | Earlier stance fine-tuning comparison scripts. |
| `4_c_Evaluation/` | Evaluation utilities. |
| `4_d_model_comparison_baseline/` | BERT, RoBERTa, PyABSA, Mistral, GPT, and Gemini baseline/final-run comparison code. |
| `4_e_multiseed_comparison/` | Reviewer-facing multi-seed comparison workflow. |
| `4_f_temporal/` | Temporal train/test split and temporal LoRA evaluation workflow. |
| `finetuned_keyword_metric_comparison/` | Per-keyword metric aggregation for fine-tuned models. |
| `mcnemar_utils.py` | Shared McNemar test helpers. |

Large model weights, checkpoints, adapters, caches, and logs are ignored. Keep
scripts, compact metrics, plots, and README files in Git.
