# Multi-Seed Model Comparison

Runs BERT, RoBERTa, PyABSA, and Mistral LoRA across fixed train/dev/test
splits for at least three random seeds. All labels are normalized to the
3-class setup: `For`, `Against`, `Neutral`.

Default run:

```bash
cd /scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/4_finetuning/4_e_multiseed_comparison
python run_multiseed_comparison.py --create-env --gpus 0,1,2,3
```

To run the four models for each seed concurrently across GPUs 0-3:

```bash
python run_multiseed_comparison.py --create-env --resume --parallel-models --gpus 0,1,2,3 --output-root ./runs/reviewer_multiseed_3seeds
```

Outputs are written to a timestamped directory under `runs/`. The old LoRA
adapter at `../../../lora_adaptor/out_train_final/mistral7b_stance_lora` is not
used or overwritten.

The main summary files are:

- `metrics_by_seed.csv`
- `metrics_summary.csv`
- `metrics_summary.md`
