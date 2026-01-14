import csv
import json
import os
import glob
from collections import defaultdict

# Paths
SOURCE_DIR = '/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/3_manual_annotation/no repetitions before finetuning'
JSON_DIR = '/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/4_finetuning/4_a_DataProcessing/data_formatting/jsons'
TRAIN_PATH = '/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/4_finetuning/4_a_DataProcessing/data_formatting/master_train.csv'
TEST_PATH = '/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/4_finetuning/4_a_DataProcessing/data_formatting/master_test.csv'

def normalize(text):
    if not text:
        return ""
    return str(text).strip()[:200]  # Use first 200 chars as robust key

def load_source_counts():
    """Load tweets from source CSVs."""
    source_stats = defaultdict(int)
    all_source_tweets = set()
    
    files = glob.glob(os.path.join(SOURCE_DIR, "*.csv"))
    print(f"Found {len(files)} source files.")
    
    for fpath in sorted(files):
        fname = os.path.basename(fpath)
        keyword = fname.replace('.csv', '')
        count = 0
        try:
            with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Try 'tweet' or 'text' or 'statement'
                    tweet = row.get('tweet') or row.get('text') or row.get('statement')
                    if tweet:
                        norm = normalize(tweet)
                        if norm:
                            source_stats[keyword] += 1
                            all_source_tweets.add(norm)
                            count += 1
        except Exception as e:
            print(f"Error reading {fname}: {e}")
            
    return source_stats, all_source_tweets

def load_dest_counts():
    """Load tweets from JSONs, Train, and Test."""
    json_tweets = set()
    train_tweets = set()
    test_tweets = set()
    
    # 1. JSONs
    json_files = glob.glob(os.path.join(JSON_DIR, "kyra_*_stance.json"))
    for fpath in json_files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
                for item in data:
                    t = item.get('statement') or item.get('tweet')
                    if t:
                        json_tweets.add(normalize(t))
        except Exception as e:
            print(f"Error reading JSON {fpath}: {e}")

    # 2. Train
    try:
        with open(TRAIN_PATH, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = row.get('tweet')
                if t:
                    train_tweets.add(normalize(t))
    except Exception as e:
        print(f"Error reading Train: {e}")

    # 3. Test
    try:
        with open(TEST_PATH, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = row.get('tweet')
                if t:
                    test_tweets.add(normalize(t))
    except Exception as e:
        print(f"Error reading Test: {e}")

    return json_tweets, train_tweets, test_tweets

def main():
    print("--- Final Count Verification ---")
    
    # 1. Source Counts (Rows)
    files = glob.glob(os.path.join(SOURCE_DIR, "*.csv"))
    source_total = 0
    print(f"\n1. Source Data ('no repetitions' folder):")
    for fpath in sorted(files):
        with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            rows = list(reader)
            # Subtract header
            count = max(0, len(rows) - 1)
            source_total += count
            # print(f"  {os.path.basename(fpath):<25}: {count}")
    
    print(f"  {'TOTAL SOURCE ROWS':<25}: {source_total}")

    # 2. Destination Counts
    print(f"\n2. Destination Data (Finetuning Set):")
    
    # JSONs
    json_count = 0
    json_files = glob.glob(os.path.join(JSON_DIR, "kyra_*_stance.json"))
    for fpath in json_files:
        with open(fpath, 'r') as f:
            json_count += len(json.load(f))
    print(f"  {'JSON Few-shot Examples':<25}: {json_count}")

    # Train
    with open(TRAIN_PATH, 'r') as f:
        reader = csv.reader(f)
        # Convert to list to count logical rows, -1 for header
        train_count = max(0, len(list(reader)) - 1)
    print(f"  {'Master Train Rows':<25}: {train_count}")

    # Test
    with open(TEST_PATH, 'r') as f:
        reader = csv.reader(f)
        test_count = max(0, len(list(reader)) - 1)
    print(f"  {'Master Test Rows':<25}: {test_count}")
    
    dest_total = json_count + train_count + test_count
    print(f"  {'TOTAL DESTINATION ROWS':<25}: {dest_total}")
    
    print("\n" + "="*40)
    print(f"FINAL RESULT: {source_total} vs {dest_total}")
    if source_total == dest_total:
        print("✅ MATCH: Exact tweet count verified.")
    else:
        print("❌ MISMATCH: Counts do not equal.")
    print("="*40)

if __name__ == "__main__":
    main()
