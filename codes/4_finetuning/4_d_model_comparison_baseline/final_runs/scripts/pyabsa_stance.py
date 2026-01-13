#!/usr/bin/env python3
"""
PyABSA Stance Classification for Final Runs
Trains PyABSA APC model on master_train.csv, evaluates on master_test.csv.
Uses ONLY PyABSA - no fallback to other models.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
import re
from tqdm import tqdm


# Paths
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
LOGS_DIR = SCRIPT_DIR.parent / "logs"
MODEL_DIR = SCRIPT_DIR.parent / "models" / "pyabsa"
DATASET_DIR = SCRIPT_DIR.parent / "pyabsa_dataset"


def setup_logging(log_file: Path):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def normalize_stance(stance: str) -> str:
    """Normalize stance labels."""
    if pd.isna(stance):
        return 'Neutral'
    s = str(stance).lower().strip()
    s = s.replace('favour', 'for').replace('favor', 'for').replace('nuetral', 'neutral')
    if s in ['for', 'positive']:
        return 'For'
    elif s in ['against', 'negative']:
        return 'Against'
    else:
        return 'Neutral'


def stance_to_polarity(stance: str) -> str:
    """Map stance to PyABSA polarity (-1, 0, 1)."""
    if stance == 'For':
        return '1'  # Positive
    elif stance == 'Against':
        return '-1'  # Negative
    else:
        return '0'  # Neutral


def polarity_to_stance(polarity) -> str:
    """Map PyABSA polarity/sentiment back to stance labels."""
    if isinstance(polarity, (int, np.integer)):
        if polarity == 1:
            return 'For'
        elif polarity == -1 or polarity == 0 and polarity != 0:
            return 'Against'
        elif polarity == 0:
            return 'Neutral'
        # Handle other numeric representations
        if polarity == 2:  # Some models use 0,1,2
            return 'For'
        elif polarity == 0:
            return 'Against'
        else:
            return 'Neutral'
    
    # String-based
    pol_str = str(polarity).lower()
    if 'positive' in pol_str or polarity == '1':
        return 'For'
    elif 'negative' in pol_str or polarity == '-1':
        return 'Against'
    else:
        return 'Neutral'


def create_pyabsa_dataset(df: pd.DataFrame, output_dir: Path, split_name: str, logger):
    """
    Create PyABSA-compatible dataset files for APC task.
    PyABSA expects 3-line format per sample:
    Line 1: text with $T$ marker where aspect goes
    Line 2: aspect term
    Line 3: polarity label (-1=Negative, 0=Neutral, 1=Positive)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    lines = []
    for _, row in df.iterrows():
        text = str(row['tweet']).replace('\n', ' ').replace('\r', ' ')
        aspect = str(row['keyword'])
        stance = row['stance']
        
        # Map stance to numeric polarity
        polarity = stance_to_polarity(stance)
        
        # Replace aspect in text with $T$ marker (case-insensitive)
        pattern = re.compile(re.escape(aspect), re.IGNORECASE)
        if pattern.search(text):
            text_with_marker = pattern.sub('$T$', text, count=1)
        else:
            # If aspect not in text, append it with marker
            text_with_marker = f"{text} $T$"
        
        # PyABSA 3-line format
        lines.append(text_with_marker)
        lines.append(aspect)
        lines.append(polarity)
    
    # Write to file - PyABSA expects: {dataset_name}.{split}.dat.apc
    output_file = output_dir / f"stance.{split_name}.dat.apc"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"  Created {output_file} with {len(df)} samples ({len(lines)} lines)")
    return output_file


def train_pyabsa_model(dataset_dir: Path, model_dir: Path, logger, epochs=5):
    """Train PyABSA APC model on custom dataset."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("Training PyABSA Model")
    logger.info("=" * 60)
    
    try:
        from pyabsa import AspectPolarityClassification as APC
    except ImportError as e:
        logger.error(f"PyABSA import error: {e}")
        logger.error("Please ensure PyABSA is installed: pip install pyabsa")
        sys.exit(1)
    
    try:
        # Get default config for English
        config = APC.APCConfigManager.get_apc_config_english()
        
        # Configure training
        config.model = APC.APCModelList.FAST_LCF_BERT
        config.pretrained_bert = 'bert-base-uncased'
        config.num_epoch = epochs
        config.batch_size = 16
        config.learning_rate = 2e-5
        config.l2reg = 1e-5
        config.max_seq_len = 256
        config.dropout = 0.1
        config.seed = 42
        config.log_step = 50
        config.patience = 3
        config.save_mode = 1  # Save best model
        
        logger.info(f"  Model: {config.model}")
        logger.info(f"  Pretrained: {config.pretrained_bert}")
        logger.info(f"  Epochs: {config.num_epoch}")
        logger.info(f"  Batch size: {config.batch_size}")
        
        # Train
        logger.info("")
        logger.info("Starting training...")
        trainer = APC.APCTrainer(
            config=config,
            dataset=str(dataset_dir),
            checkpoint_save_mode=1,
            auto_device=True,
            path_to_save=str(model_dir)
        )
        
        logger.info(f"✓ Model trained and saved to: {model_dir}")
        return trainer
        
    except Exception as e:
        import traceback
        logger.error(f"Training failed: {e}")
        logger.error("Full traceback:")
        traceback.print_exc()
        sys.exit(1)


def run_inference(model_dir: Path, test_df: pd.DataFrame, logger) -> pd.DataFrame:
    """Run inference using the trained PyABSA model."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("Running PyABSA Inference")
    logger.info("=" * 60)
    
    try:
        from pyabsa import AspectPolarityClassification as APC
    except ImportError as e:
        logger.error(f"PyABSA import error: {e}")
        sys.exit(1)
    
    # Find the trained checkpoint
    checkpoint_path = None
    logger.info(f"Looking for checkpoints in: {model_dir}")
    
    if model_dir.exists():
        for item in model_dir.iterdir():
            if item.is_dir():
                logger.info(f"  Found directory: {item.name}")
                checkpoint_path = item
                break
    
    if checkpoint_path is None:
        # Try the model_dir itself
        checkpoint_path = model_dir
    
    logger.info(f"Loading model from: {checkpoint_path}")
    
    try:
        classifier = APC.SentimentClassifier(checkpoint=str(checkpoint_path), auto_device=True)
    except Exception as e:
        logger.error(f"Failed to load trained model: {e}")
        logger.error("Ensure training completed successfully.")
        sys.exit(1)
    
    predictions = []
    logger.info(f"Processing {len(test_df)} samples...")
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Classifying"):
        tweet = str(row['tweet'])
        keyword = str(row['keyword'])
        original_stance = row['stance']
        
        try:
            # PyABSA format: text with aspect marked using [B-ASP] and [E-ASP]
            formatted_input = f"{tweet} [B-ASP]{keyword}[E-ASP]"
            result = classifier.predict(formatted_input, print_result=False)
            
            # Extract sentiment from result
            if hasattr(result, 'sentiment'):
                sentiment = result.sentiment
                if isinstance(sentiment, list) and len(sentiment) > 0:
                    sentiment = sentiment[0]
            elif isinstance(result, dict):
                sentiment = result.get('sentiment', 'Neutral')
                if isinstance(sentiment, list) and len(sentiment) > 0:
                    sentiment = sentiment[0]
            elif isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict):
                    sentiment = result[0].get('sentiment', 'Neutral')
                else:
                    sentiment = str(result[0])
            else:
                sentiment = str(result)
            
            stance = polarity_to_stance(sentiment)
            
        except Exception as e:
            # Log but don't fail - default to Neutral
            logger.warning(f"Error predicting sample {idx}: {e}")
            stance = 'Neutral'
        
        predictions.append({
            'tweet': tweet,
            'keyword': keyword,
            'original_stance': original_stance,
            'pyabsa_prediction': stance
        })
    
    return pd.DataFrame(predictions)


def main():
    parser = argparse.ArgumentParser(description='PyABSA Stance Classification')
    parser.add_argument('--train-csv', type=str, required=True, help='Path to train CSV')
    parser.add_argument('--test-csv', type=str, required=True, help='Path to test CSV')
    parser.add_argument('--output-dir', type=str, default=str(RESULTS_DIR), help='Output directory')
    parser.add_argument('--log-file', type=str, default=str(LOGS_DIR / 'pyabsa.log'), help='Log file path')
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('--skip-train', action='store_true', help='Skip training and use existing model')
    args = parser.parse_args()
    
    # Setup logging
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_path)
    
    logger.info("=" * 60)
    logger.info("PyABSA Stance Classification")
    logger.info("=" * 60)
    
    # Create directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"Loading train data from: {args.train_csv}")
    train_df = pd.read_csv(args.train_csv)
    train_df['stance'] = train_df['stance'].apply(normalize_stance)
    logger.info(f"  Train samples: {len(train_df)}, Unique tweets: {train_df['tweet'].nunique()}")
    
    logger.info(f"Loading test data from: {args.test_csv}")
    test_df = pd.read_csv(args.test_csv)
    test_df['stance'] = test_df['stance'].apply(normalize_stance)
    logger.info(f"  Test samples: {len(test_df)}, Unique tweets: {test_df['tweet'].nunique()}")
    
    # Create PyABSA dataset files
    logger.info("")
    logger.info("Creating PyABSA dataset files...")
    
    # Split train for validation (90/10)
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(
        train_df, test_size=0.1, random_state=42, stratify=train_df['stance']
    )
    
    create_pyabsa_dataset(train_data, DATASET_DIR, "train", logger)
    create_pyabsa_dataset(val_data, DATASET_DIR, "test", logger)
    
    # Train model (unless skipped)
    if not args.skip_train:
        train_pyabsa_model(DATASET_DIR, MODEL_DIR, logger, epochs=args.epochs)
    else:
        logger.info("Skipping training, using existing model...")
    
    # Run inference
    predictions_df = run_inference(MODEL_DIR, test_df, logger)
    
    # Save predictions
    output_path = output_dir / "pyabsa_predictions.csv"
    predictions_df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved predictions to: {output_path}")
    
    # Report accuracy
    correct = (predictions_df['original_stance'] == predictions_df['pyabsa_prediction']).sum()
    total = len(predictions_df)
    accuracy = correct / total * 100
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("PyABSA Results")
    logger.info("=" * 60)
    logger.info(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    logger.info(f"\nGround truth distribution:\n{predictions_df['original_stance'].value_counts().to_string()}")
    logger.info(f"\nPrediction distribution:\n{predictions_df['pyabsa_prediction'].value_counts().to_string()}")
    logger.info("✓ PyABSA training and inference complete!")


if __name__ == "__main__":
    main()
