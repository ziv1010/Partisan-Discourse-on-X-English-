#!/usr/bin/env python3
"""
Prepare Extracted Data for Stance Inference

The keyword_extractor_v2.py outputs CSVs with 'english_text' column,
but finetune_stance.py expects 'tweet' column.

This script renames 'english_text' -> 'tweet' for stance inference.
"""

import pandas as pd
import sys
from pathlib import Path

# Input/Output paths
INPUT_CSV = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/hindi_codes/extracted_by_keyword/extracted_ALL_37keywords_233580rows.csv"
OUTPUT_CSV = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/hindi_codes/extracted_by_keyword/extracted_for_stance_inference.csv"

print("=" * 70)
print("PREPARE EXTRACTED DATA FOR STANCE INFERENCE")
print("=" * 70)

# Load CSV
print(f"\nLoading: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV, low_memory=False)
print(f"Loaded {len(df):,} rows")
print(f"Original columns: {df.columns.tolist()}")

# Rename english_text -> tweet for stance inference
if 'english_text' in df.columns:
    df = df.rename(columns={'english_text': 'tweet'})
    print(f"\nRenamed 'english_text' -> 'tweet'")

# Ensure we have required columns: tweet, keyword
required_cols = ['tweet', 'keyword']
for col in required_cols:
    if col not in df.columns:
        print(f"ERROR: Required column '{col}' not found!")
        sys.exit(1)

print(f"\nFinal columns: {df.columns.tolist()}")

# Save
print(f"\nSaving: {OUTPUT_CSV}")
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(df):,} rows")

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)
print(f"\nOutput file ready for stance inference: {OUTPUT_CSV}")
