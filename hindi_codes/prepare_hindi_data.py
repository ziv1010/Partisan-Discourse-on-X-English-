#!/usr/bin/env python3
"""
Prepare Hindi Data for Keyword Extraction and Stance Inference

This script renames columns in the translated Hindi dataset to match 
the expected column names for:
1. keyword_extractor_v2.py - uses: english_text for keyword matching, tweet_label for labels
2. parallel_inference.py (finetune_stance.py) - expects: tweet, keyword

Current CSV columns:
- tweet, tweet_label, devanagari_ratio, script, en_text, translation_engine, cosine_before_after

Output CSV columns:
- hindi_tweet, tweet_label, label_score, script, english_text, translation_model, translation_score

For stance inference after keyword extraction, the extracted CSV will have 'english_text' 
which needs to be the 'tweet' column expected by finetune_stance.py.
"""

import pandas as pd
from pathlib import Path

# Paths
INPUT_CSV = "/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/hindi_codes/hindi_data/full_hi_en.csv"
OUTPUT_DIR = Path("/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/hindi_codes/hindi_data")

# Output file - will be saved WITH HEADERS
OUTPUT_CSV = OUTPUT_DIR / "full_hi_en_prepared.csv"

print("=" * 70)
print("PREPARE HINDI DATA FOR KEYWORD EXTRACTION AND STANCE INFERENCE")
print("=" * 70)

# Load CSV
print(f"\nLoading: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV, low_memory=False)
print(f"Loaded {len(df):,} rows")
print(f"Original columns: {df.columns.tolist()}")

# Rename columns
# Current: tweet, tweet_label, devanagari_ratio, script, en_text, translation_engine, cosine_before_after
# Target: hindi_tweet, tweet_label, label_score, script, english_text, translation_model, translation_score

column_mapping = {
    'tweet': 'hindi_tweet',                   # Original Hindi text -> hindi_tweet (user request)
    'tweet_label': 'tweet_label',             # Keep as is
    'devanagari_ratio': 'label_score',        # Map to label_score
    'script': 'script',                       # Keep as is  
    'en_text': 'english_text',                # English translation -> english_text
    'translation_engine': 'translation_model',# Map to translation_model
    'cosine_before_after': 'translation_score'# Map to translation_score
}

# Apply renaming
df_renamed = df.rename(columns=column_mapping)
print(f"\nRenamed columns: {df_renamed.columns.tolist()}")

# Save WITH HEADERS
print(f"\nSaving: {OUTPUT_CSV}")
df_renamed.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(df_renamed):,} rows")

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)
print(f"\nOutput file: {OUTPUT_CSV}")
print(f"\nColumn mapping:")
for old, new in column_mapping.items():
    print(f"  {old:25} -> {new}")

print(f"""
Next steps:
1. Run keyword_extractor_v2.py (it will use full_hi_en_prepared.csv)
2. Run parallel_inference.py on the extracted data
""")
