# Multilingual Translation Pipeline

This directory contains the high-accuracy translation workflow used to convert a mixed English, Hindi-script, and Roman Hindi/Hinglish X dataset into English while preserving enough intermediate outputs to audit the routing and translation decisions.

The pipeline implements the methodological choice you described as the stronger revision:

1. Detect English vs native Indic script vs Roman Hindi/Hinglish.
2. Leave English unchanged.
3. Send native Indic-script tweets directly to Hindi-to-English translation with the correct NLLB source language when available.
4. Normalize Roman Hindi/Hinglish into Devanagari first, while preserving obvious English tokens where possible.
5. Translate the normalized Roman branch to English.
6. Merge all branches back into a single row-ordered English dataset.

## Environment

The pipeline runs in the `micromamba` environment `x_translate`.

Core packages installed there include:

- `pandas`, `numpy`, `pyarrow`, `tqdm`
- `fasttext`
- `transformers==4.48.3`
- `torch==2.5.1`
- `fairseq==0.12.2`
- `sentencepiece`
- `indic_nlp_library`
- `wordfreq`
- local install of `IndicXlit`

The environment was chosen because:

- `IndicLID` needs both fastText and BERT-based language identification support.
- `IndicXlit` provides transliteration into Devanagari.
- NLLB translation runs reliably with `transformers==4.48.3` in this setup.

## Main Scripts

### [prepare_translation_routes.py](/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Additional_Steps/Translating/prepare_translation_routes.py)

This is the routing and normalization stage.

What it does:

- Streams the input CSV in chunks.
- Runs `IndicLID` to classify each tweet.
- Splits the full dataset into three branch files:
  - `english_tweets.csv`
  - `hindi_script_tweets.csv`
  - `hinglish_roman_tweets.csv`
- Preserves the original text in `tweet_original`.
- Adds `row_id` so the final merge can restore original ordering.
- Writes routing metadata:
  - `translation_route`
  - `lid_lang`
  - `lid_score`
  - `lid_model`
  - `xlit_lang`
  - `nllb_src_lang`

Important implementation details:

- Any tweet containing Indic script characters is routed to the native-script branch even if it also contains Latin text, hashtags, or links.
- Roman Hindi/Hinglish is not treated as a direct translation problem. It is treated as a normalization problem.
- Selective transliteration is used so obvious English tokens are more likely to stay English while Romanized Hindi tokens are normalized into Devanagari.
- Roman non-English tweets are normalized to Hindi-script and then assigned `nllb_src_lang=hin_Deva` for translation.
- Native-script tweets keep the predicted NLLB source language when available, for example `mar_Deva` or `hin_Deva`.

### [full_run_hindi_pipeline.py](/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Additional_Steps/Translating/full_run_hindi_pipeline.py)

This is the translation stage.

What it does:

- Translates a CSV to English using NLLB.
- Can use a fixed source language or a per-row source language from `--src-lang-col`.
- Preserves all original columns and appends:
  - `devanagari_ratio`
  - `script`
  - `en_text`
  - `translation_engine`
  - `cosine_before_after`

Important implementation details:

- The script now supports per-row source language routing, which matters because your routed native branch is not only Hindi. It also includes other Indic languages detected in Devanagari, such as Marathi.
- NLLB alias normalization is built in so common label mismatches still resolve correctly.
- The NLLB loader was patched to avoid the meta-tensor device-map issue by loading the model without `device_map` and then moving it explicitly to the target device.
- The script can emit low-cosine rows for later inspection.

### [merge_translation_outputs.py](/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Additional_Steps/Translating/merge_translation_outputs.py)

This is the final merge stage.

What it does:

- Reads:
  - the English passthrough branch
  - the translated native-script branch
  - the translated Roman/Hinglish branch
- Merges them back together in `row_id` order.
- Produces the final English dataset.

For the English branch:

- `en_text` is set equal to the original tweet text.
- `translation_engine` is set to `unchanged`.

### [run_full_multilingual_translation.sh](/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Additional_Steps/Translating/run_full_multilingual_translation.sh)

This is the full production runner.

What it does:

1. Runs routing.
2. Launches native-script translation.
3. Launches Roman/Hinglish translation.
4. Waits for both branches.
5. Runs the merge.

It writes everything into a dated run directory under `Data/`.

### [resume_remaining_translation.sh](/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Additional_Steps/Translating/resume_remaining_translation.sh)

This is the recovery runner for interrupted or partially failed runs.

What it does:

1. Reuses an existing run directory.
2. Reads the routing log to recover expected branch totals.
3. Waits for an already-running native translation branch if one is active.
4. Reruns the Roman/Hinglish branch if it failed or is incomplete.
5. Validates that translated totals match routed totals.
6. Runs the final merge.

This script was added because the first Roman/Hinglish translation attempt failed from CUDA OOM on a crowded GPU.

## Current Production Run

The active production run is:

- [full_multilingual_en_20260403_detached](/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Additional_Steps/Translating/Data/full_multilingual_en_20260403_detached)

Key subdirectories:

- [routed](/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Additional_Steps/Translating/Data/full_multilingual_en_20260403_detached/routed)
- [translated](/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Additional_Steps/Translating/Data/full_multilingual_en_20260403_detached/translated)
- [logs](/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Additional_Steps/Translating/Data/full_multilingual_en_20260403_detached/logs)

Routing finished for:

- `7,115,963` total rows
- `3,006,222` English rows
- `4,078,560` native-script rows
- `31,181` Roman/Hinglish rows

The main log files for this run are:

- [01_route.log](/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Additional_Steps/Translating/Data/full_multilingual_en_20260403_detached/logs/01_route.log)
- [02_translate_native.log](/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Additional_Steps/Translating/Data/full_multilingual_en_20260403_detached/logs/02_translate_native.log)
- [03_translate_roman.log](/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Additional_Steps/Translating/Data/full_multilingual_en_20260403_detached/logs/03_translate_roman.log)
- [04_merge.log](/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Additional_Steps/Translating/Data/full_multilingual_en_20260403_detached/logs/04_merge.log)

Expected final merged file:

- [retweet_data_combined_labeled_by_retweet_en.csv](/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Additional_Steps/Translating/Data/full_multilingual_en_20260403_detached/retweet_data_combined_labeled_by_retweet_en.csv)

## Exact Data Flow

### Step 1: Input

The source dataset is:

- [retweet_data_combined_labeled_by_retweet.csv](/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Additional_Steps/Translating/Data/retweet_data_combined_labeled_by_retweet.csv)

The main text column used by the pipeline is:

- `tweet`

### Step 2: Route each row

For each row:

- The original text is preserved in `tweet_original`.
- A sequential `row_id` is assigned.
- Language/script routing decides which branch the row belongs to.

Branch rules:

- `english`
  - Row is kept unchanged.
- `hindi_script`
  - Row already contains native Indic script.
  - `nllb_src_lang` is set from IndicLID where possible.
- `hinglish_roman`
  - Row is Roman Hindi/Hinglish-like.
  - Text is selectively transliterated to Devanagari.
  - `nllb_src_lang` is set to `hin_Deva`.

### Step 3: Translate native Indic script branch

The native branch file:

- [hindi_script_tweets.csv](/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Additional_Steps/Translating/Data/full_multilingual_en_20260403_detached/routed/hindi_script_tweets.csv)

is translated with per-row NLLB source language routing into:

- [hindi_script_en.csv](/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Additional_Steps/Translating/Data/full_multilingual_en_20260403_detached/translated/hindi_script_en.csv)

### Step 4: Translate Roman/Hinglish branch

The Roman branch file:

- [hinglish_roman_tweets.csv](/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Additional_Steps/Translating/Data/full_multilingual_en_20260403_detached/routed/hinglish_roman_tweets.csv)

is normalized to Devanagari during routing and then translated into:

- [hinglish_roman_en.csv](/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Additional_Steps/Translating/Data/full_multilingual_en_20260403_detached/translated/hinglish_roman_en.csv)

### Step 5: Merge all branches

The merge stage restores original row order by `row_id` and writes:

- the unchanged English rows
- the translated native-script rows
- the translated Roman/Hinglish rows

into one final English CSV.

## Quality-Control Outputs

The translation script computes cosine similarity between the source and translated text embeddings.

These are used to:

- characterize translation confidence at scale
- produce low-cosine audit files for closer review

Files produced by the translator include:

- `hindi_script_lowcos.csv`
- `hinglish_roman_lowcos.csv`

These are intended for inspection, not for primary analysis.

## Monitoring the Run

Two `tmux` sessions are currently relevant:

- `x_translate_full_en_20260403`
  - original full runner
- `x_translate_resume_remaining_20260403`
  - recovery runner that handles the failed Roman branch and final merge

Useful commands:

```bash
tmux ls
tmux capture-pane -pt x_translate_full_en_20260403
tmux capture-pane -pt x_translate_resume_remaining_20260403
tail -f /scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Additional_Steps/Translating/Data/full_multilingual_en_20260403_detached/logs/02_translate_native.log
tail -f /scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Additional_Steps/Translating/Data/full_multilingual_en_20260403_detached/logs/03_translate_roman.log
```

## Re-running from Scratch

To run a fresh full job:

```bash
cd /scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Additional_Steps/Translating
RUN_TAG=my_new_run ./run_full_multilingual_translation.sh \
  /scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Additional_Steps/Translating/Data/retweet_data_combined_labeled_by_retweet.csv
```

## Resuming a Partial Run

To continue an existing run directory after a failed branch:

```bash
cd /scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Additional_Steps/Translating
ROMAN_GPU=0 ROMAN_BATCH_SIZE=16 ./resume_remaining_translation.sh \
  /scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Additional_Steps/Translating/Data/full_multilingual_en_20260403_detached
```

## Notebook

The companion notebook for exploring routed, translated, and merged outputs is:

- [translation_pipeline_review.ipynb](/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/Additional_Steps/Translating/translation_pipeline_review.ipynb)

The notebook is designed to:

- inspect route distributions
- sample tweets from each branch
- compare original vs normalized vs translated text
- inspect low-cosine examples
- look up rows by `row_id`
- search tweets by text across intermediate outputs
