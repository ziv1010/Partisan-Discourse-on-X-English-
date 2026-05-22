# retweet_data_combined_labeled_by_retweet_en_with_hinglish_bridge split bundle

This bundle contains a row-safe split of `retweet_data_combined_labeled_by_retweet_en_with_hinglish_bridge.csv`.

## Why the split uses Python instead of shell tools

The source CSV contains embedded newlines inside quoted tweet text. Because of that, line-based tools such as `split`, `sed`, or `wc -l` do not reflect true CSV row boundaries. The provided scripts use Python's CSV parser so each output file preserves valid records and column structure.

## Bundle contents

- `parts/`: 8 split CSV files
- `split_csv_parts.py`: script used to create the parts
- `append_csv_parts.py`: script to recombine the parts into one CSV
- `metadata.json`: exact row counts and filenames for each part

## Split summary

- Source file: `retweet_data_combined_labeled_by_retweet_en_with_hinglish_bridge.csv`
- Data rows: `7,115,963`
- Columns: `23`
- Rows per full part: `1,000,000`
- Part count: `8`
- Final part rows: `115,963`

## Recombine the parts

From this folder, run:

```bash
python3 append_csv_parts.py
```

That writes:

```text
./retweet_data_combined_labeled_by_retweet_en_with_hinglish_bridge_recombined.csv
```

To choose a custom output path:

```bash
python3 append_csv_parts.py ./parts /path/to/output.csv --overwrite
```

## Re-run the split

If you need to recreate the parts from the original CSV:

```bash
python3 split_csv_parts.py /path/to/source.csv ./parts --rows-per-part 1000000
```

Only run the split script into an empty `parts/` directory.

## Validation

The append script was executed successfully on 2026-04-06 and rebuilt a single CSV with `7,115,963` data rows from all 8 parts. The temporary verification output was then removed so it is not included in this share bundle.
