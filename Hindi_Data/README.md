# Hindi Data Evaluation Scripts

This folder contains standalone scripts used for Hindi translation evaluation,
drift inspection, and multilingual evaluation reruns.

| Script | Purpose |
| --- | --- |
| `full_run_hindi_pipeline.py` | Runs the Hindi translation workflow for evaluation inputs. |
| `translate_eval_5000.py` | Translates the evaluation sample. |
| `rerun_eval5000_multilingual.py` | Re-runs multilingual evaluation on the translated sample. |
| `make_eval_5000.py` | Builds the evaluation subset. |
| `analyze_drift.py` and `plot_umap_drift_with_legend.py` | Inspect and visualize embedding drift. |
| `breakdown.py` | Produces simple breakdowns of Hindi evaluation data. |

Large intermediate data files are not committed; restore local inputs before
running the scripts.
