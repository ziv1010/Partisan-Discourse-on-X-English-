# Multilingual Translation Entry Point

This directory contains the routing, translation, audit, and merge scripts used
to convert mixed English, Hindi-script, and Roman Hindi/Hinglish tweets into an
English analysis dataset.

Start with the detailed workflow notes in
`README_multilingual_translation.md`. The main executable stages are:

| Script | Purpose |
| --- | --- |
| `prepare_translation_routes.py` | Classifies tweet language/script and writes branch-specific route files. |
| `full_run_hindi_pipeline.py` | Runs NLLB-based translation for routed native-script or Roman/Hinglish branches. |
| `merge_translation_outputs.py` | Merges English passthrough and translated branches back into row order. |
| `run_full_multilingual_translation.sh` | Production runner for routing, translation, and merge. |
| `resume_remaining_translation.sh` | Recovery runner for partially completed translation jobs. |
| `assemble_native_multigpu_outputs.py` | Combines multi-GPU shard outputs. |

The `Data/` and `models/` subtrees contain heavyweight local artifacts and are
ignored except for source files and README-style documentation.
