# Fine-Tuned Model Per-Keyword Metrics

Generated combined per-keyword accuracy and macro-F1 files for the two fine-tuned Mistral LoRA models.

## Outputs

- `finetuned_models_per_keyword_accuracy_f1_long.csv`: one row per model-keyword.
- `finetuned_models_per_keyword_accuracy_f1_wide.csv`: one row per keyword, with normal and temporal columns side by side.

## Sources

- Normal: `/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/4_finetuning/4_b_TestingComparison/results/evaluation/per_keyword_metrics.csv`
- Temporal: `/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/4_finetuning/4_f_temporal/runs/temporal_lora_2020_2021_test_2022_2023_gpus_5_7/evaluation/per_keyword_metrics.csv`

## Row Accounting

- Normal fine-tuned model: 264 evaluated samples across 15 keywords.
- Temporal fine-tuned model: 684 evaluated samples across 35 keywords.

Note: the normal evaluation file contains only `count`, `accuracy`, and `f1_macro` per keyword. The temporal evaluation includes additional fields such as weighted F1, precision, recall, and gold/predicted class counts; these are preserved in the long CSV.

Regenerate with:

```bash
python3 extract_finetuned_keyword_metrics.py
```
