"""
Data Preprocessor for ABSA Baseline Comparison
Loads train/test CSVs, normalizes stance labels, and applies text cleaning.
Supports multiple cleaning levels: basic, moderate, aggressive.
"""

import os
import re
import string
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# NLTK for stopword removal
try:
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
    STOP_WORDS = set(stopwords.words('english'))
except ImportError:
    NLTK_AVAILABLE = False
    STOP_WORDS = set()
    print("Warning: NLTK not available, stopword removal disabled")

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "4_a_DataProcessing" / "data_formatting"
OUTPUT_DIR = BASE_DIR / "processed_data"


def normalize_stance(stance: str) -> str:
    """
    Normalize stance labels to 3 classes: For, Against, Neutral
    Handles case variations and typos in original data.
    """
    if pd.isna(stance):
        return "Neutral"
    
    stance_lower = str(stance).lower().strip()
    
    # Map to standard labels
    if stance_lower in ["for", "favour"]:
        return "For"
    elif stance_lower in ["against"]:
        return "Against"
    elif stance_lower in ["neutral", "nuetral"]:
        return "Neutral"
    else:
        return "Neutral"


def clean_text_basic(text: str) -> str:
    """
    Basic text cleaning:
    - Remove URLs
    - Remove extra whitespace
    - Keep hashtags and mentions
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def clean_text_moderate(text: str) -> str:
    """
    Moderate text cleaning:
    - All basic cleaning
    - Remove special characters (keep alphanumeric, @, #)
    - Lowercase
    """
    text = clean_text_basic(text)
    
    if not text:
        return ""
    
    # Remove special characters except @ and #
    text = re.sub(r'[^\w\s@#]', ' ', text)
    
    # Remove extra whitespace again
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Lowercase
    text = text.lower()
    
    return text


def clean_text_aggressive(text: str) -> str:
    """
    Aggressive text cleaning:
    - All moderate cleaning
    - Remove hashtags and mentions
    - Remove stopwords
    - Remove numbers
    """
    text = clean_text_moderate(text)
    
    if not text:
        return ""
    
    # Remove hashtags and mentions
    text = re.sub(r'[@#]\w+', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove stopwords if available
    if NLTK_AVAILABLE and STOP_WORDS:
        try:
            tokens = word_tokenize(text)
            tokens = [t for t in tokens if t.lower() not in STOP_WORDS and len(t) > 1]
            text = ' '.join(tokens)
        except Exception:
            pass  # Fall back to keeping text as is
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def get_cleaner(cleaning_level: str):
    """Get the appropriate cleaning function."""
    cleaners = {
        'basic': clean_text_basic,
        'moderate': clean_text_moderate,
        'aggressive': clean_text_aggressive,
        'none': lambda x: str(x) if pd.notna(x) else ""
    }
    return cleaners.get(cleaning_level, clean_text_basic)


def process_dataframe(df: pd.DataFrame, cleaning_level: str = 'basic') -> pd.DataFrame:
    """
    Process a dataframe:
    - Extract relevant columns
    - Normalize stance labels
    - Clean tweet text based on cleaning level
    """
    cleaner = get_cleaner(cleaning_level)
    
    # Extract relevant columns
    processed_df = pd.DataFrame()
    
    processed_df['original_tweet'] = df['tweet']
    processed_df['tweet'] = df['tweet'].apply(cleaner)
    processed_df['keyword'] = df['keyword'].str.lower().str.strip()
    processed_df['original_stance'] = df['stance']
    processed_df['stance'] = df['stance'].apply(normalize_stance)
    
    # Keep source_row for reference
    if 'source_row' in df.columns:
        processed_df['source_row'] = df['source_row']
    
    # Remove empty tweets
    processed_df = processed_df[processed_df['tweet'].str.len() > 0]
    
    return processed_df


def main():
    """Main preprocessing pipeline."""
    parser = argparse.ArgumentParser(description='Preprocess data for ABSA')
    parser.add_argument('--cleaning', type=str, default='basic',
                       choices=['none', 'basic', 'moderate', 'aggressive'],
                       help='Cleaning level: none, basic, moderate, aggressive')
    args = parser.parse_args()
    
    cleaning_level = args.cleaning
    
    print("=" * 60)
    print("ABSA Baseline - Data Preprocessing")
    print("=" * 60)
    print(f"\nCleaning level: {cleaning_level}")
    
    # Describe cleaning levels
    cleaning_info = {
        'none': 'No cleaning (raw text)',
        'basic': 'Remove URLs, extra whitespace',
        'moderate': 'Basic + remove special chars, lowercase',
        'aggressive': 'Moderate + remove stopwords, hashtags, mentions, numbers'
    }
    print(f"  → {cleaning_info[cleaning_level]}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load train data
    train_path = DATA_DIR / "master_train.csv"
    print(f"\nLoading train data from: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"  Original rows: {len(train_df)}")
    
    # Process train data
    train_processed = process_dataframe(train_df, cleaning_level)
    print(f"  Processed rows: {len(train_processed)}")
    
    # Save train data
    train_output = OUTPUT_DIR / "train_processed.csv"
    train_processed.to_csv(train_output, index=False)
    print(f"  Saved to: {train_output}")
    
    # Load test data
    test_path = DATA_DIR / "master_test.csv"
    print(f"\nLoading test data from: {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"  Original rows: {len(test_df)}")
    
    # Process test data
    test_processed = process_dataframe(test_df, cleaning_level)
    print(f"  Processed rows: {len(test_processed)}")
    
    # Save test data
    test_output = OUTPUT_DIR / "test_processed.csv"
    test_processed.to_csv(test_output, index=False)
    print(f"  Saved to: {test_output}")
    
    # Print label distribution
    print("\n" + "=" * 60)
    print("Stance Label Distribution (after normalization)")
    print("=" * 60)
    
    print("\nTrain set:")
    print(train_processed['stance'].value_counts().to_string())
    
    print("\nTest set:")
    print(test_processed['stance'].value_counts().to_string())
    
    print("\n" + "=" * 60)
    print("Unique keywords:")
    print("=" * 60)
    print(f"Train: {sorted(train_processed['keyword'].unique())}")
    print(f"Test: {sorted(test_processed['keyword'].unique())}")
    
    # Show example of cleaning
    print("\n" + "=" * 60)
    print("Cleaning Example:")
    print("=" * 60)
    sample_idx = train_processed.index[0]
    print(f"Original: {train_processed.loc[sample_idx, 'original_tweet'][:100]}...")
    print(f"Cleaned:  {train_processed.loc[sample_idx, 'tweet'][:100]}...")
    
    print("\n✓ Preprocessing complete!")
    
    return train_processed, test_processed


if __name__ == "__main__":
    main()
