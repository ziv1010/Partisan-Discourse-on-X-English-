"""
PyABSA-based Aspect-Based Sentiment Analysis with Training
Trains PyABSA on custom dataset or uses pre-trained with fallback.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Set environment to avoid TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# Paths
BASE_DIR = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / "processed_data"
RESULTS_DIR = BASE_DIR / "results"
MODEL_DIR = BASE_DIR / "pyabsa_model"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"


def map_stance_to_sentiment(stance: str) -> str:
    """Map stance labels to PyABSA sentiment format."""
    mapping = {
        'For': 'Positive',
        'Against': 'Negative',
        'Neutral': 'Neutral'
    }
    return mapping.get(stance, 'Neutral')


def map_sentiment_to_stance(sentiment_label) -> str:
    """Map sentiment/prediction back to stance labels."""
    if isinstance(sentiment_label, (int, np.integer)):
        # Numeric label (0=Negative, 1=Neutral, 2=Positive typically)
        if sentiment_label == 2 or sentiment_label == 1:  # Positive
            return 'For'
        elif sentiment_label == 0 or sentiment_label == -1:  # Negative
            return 'Against'
        else:
            return 'Neutral'
    
    sentiment_lower = str(sentiment_label).lower()
    
    if 'positive' in sentiment_lower:
        return 'For'
    elif 'negative' in sentiment_lower:
        return 'Against'
    else:
        return 'Neutral'


def create_pyabsa_dataset(df: pd.DataFrame, output_dir: Path, split_name: str):
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
        if stance == 'For':
            polarity = '1'  # Positive
        elif stance == 'Against':
            polarity = '-1'  # Negative
        else:
            polarity = '0'  # Neutral
        
        # Replace aspect in text with $T$ marker (case-insensitive)
        import re
        pattern = re.compile(re.escape(aspect), re.IGNORECASE)
        if pattern.search(text):
            text_with_marker = pattern.sub('$T$', text, count=1)
        else:
            # If aspect not in text, append it
            text_with_marker = f"{text} $T$"
        
        # PyABSA 3-line format
        lines.append(text_with_marker)
        lines.append(aspect)
        lines.append(polarity)
    
    # Write to file with dataset name prefix
    # PyABSA expects: {dataset_name}.{split}.dat
    output_file = output_dir / f"stance.{split_name}.dat"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"  Created {output_file} with {len(df)} samples ({len(lines)} lines)")
    return output_file


def train_pyabsa_model(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """Train PyABSA APC model on custom dataset."""
    print("\n" + "=" * 60)
    print("Training PyABSA Model")
    print("=" * 60)
    
    try:
        from pyabsa import AspectPolarityClassification as APC
        from pyabsa.utils.data_utils.dataset_manager import detect_infer_dataset
    except ImportError as e:
        print(f"PyABSA import error: {e}")
        return None
    
    # Create dataset files
    dataset_dir = BASE_DIR / "pyabsa_dataset"
    print("\nCreating PyABSA dataset files...")
    create_pyabsa_dataset(train_df, dataset_dir, "train")
    create_pyabsa_dataset(val_df, dataset_dir, "test")
    
    # Configure training
    print("\nConfiguring training...")
    
    try:
        # Get default config
        config = APC.APCConfigManager.get_apc_config_english()
        
        # Modify config for our use case
        config.model = APC.APCModelList.FAST_LCF_BERT
        config.pretrained_bert = 'bert-base-uncased'
        config.num_epoch = 5
        config.batch_size = 16
        config.learning_rate = 2e-5
        config.l2reg = 1e-5
        config.max_seq_len = 256
        config.dropout = 0.1
        config.seed = 42
        config.log_step = 50
        config.patience = 3
        config.save_mode = 1  # Save best model
        
        # Set paths
        config.dataset_file = str(dataset_dir)
        config.dataset_name = "stance_dataset"
        config.output_dir = str(MODEL_DIR)
        
        print(f"  Model: {config.model}")
        print(f"  Pretrained: {config.pretrained_bert}")
        print(f"  Epochs: {config.num_epoch}")
        print(f"  Batch size: {config.batch_size}")
        
        # Train
        print("\nStarting training...")
        trainer = APC.APCTrainer(
            config=config,
            dataset=str(dataset_dir),
            checkpoint_save_mode=1,
            auto_device=True
        )
        
        print(f"\n✓ Model trained and saved to: {MODEL_DIR}")
        return trainer
        
    except Exception as e:
        import traceback
        print(f"Training failed: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("\nFalling back to pre-trained model inference...")
        return None


def run_pretrained_inference(test_df: pd.DataFrame) -> pd.DataFrame:
    """Run inference using pre-trained PyABSA checkpoints."""
    print("\n" + "=" * 60)
    print("Running PyABSA Pre-trained Model Inference")
    print("=" * 60)
    
    classifier = None
    model_name = None
    
    try:
        from pyabsa import AspectPolarityClassification as APC
        
        # Try multilingual checkpoint first
        multilingual_path = CHECKPOINT_DIR / "APC_MULTILINGUAL_CHECKPOINT"
        english_path = CHECKPOINT_DIR / "APC_ENGLISH_CHECKPOINT" / "fast_lsa_t_v2_English_acc_82.21_f1_81.81"
        
        if multilingual_path.exists():
            print(f"Loading PyABSA multilingual checkpoint from: {multilingual_path}")
            try:
                classifier = APC.SentimentClassifier(checkpoint=str(multilingual_path), auto_device=True)
                model_name = "APC_MULTILINGUAL"
                print("✓ Loaded PyABSA multilingual checkpoint")
            except Exception as e:
                print(f"Failed to load multilingual checkpoint: {e}")
        
        if classifier is None and english_path.exists():
            print(f"Loading PyABSA English checkpoint from: {english_path}")
            try:
                classifier = APC.SentimentClassifier(checkpoint=str(english_path), auto_device=True)
                model_name = "APC_ENGLISH"
                print("✓ Loaded PyABSA English checkpoint")
            except Exception as e:
                print(f"Failed to load English checkpoint: {e}")
        
        if classifier is None:
            # Try to load from PyABSA hub
            print("Attempting to load PyABSA model from hub...")
            try:
                classifier = APC.SentimentClassifier(checkpoint='multilingual', auto_device=True)
                model_name = "APC_HUB_MULTILINGUAL"
                print("✓ Loaded PyABSA model from hub")
            except Exception as e:
                print(f"Failed to load from hub: {e}")
                raise RuntimeError("Could not load any PyABSA checkpoint")
                
    except Exception as e:
        print(f"PyABSA loading failed: {e}")
        print("\nERROR: Cannot proceed without PyABSA. Please ensure PyABSA is installed:")
        print("  pip install pyabsa")
        sys.exit(1)
    
    predictions = []
    
    print(f"\nProcessing {len(test_df)} samples with {model_name}...")
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Classifying"):
        tweet = str(row['tweet'])
        keyword = str(row['keyword'])
        
        try:
            # PyABSA format: text with aspect marked
            formatted_input = f"{tweet} [B-ASP]{keyword}[E-ASP]"
            result = classifier.predict(formatted_input, print_result=False)
            
            # Extract sentiment from result
            if hasattr(result, 'sentiment'):
                sentiment = result.sentiment
            elif isinstance(result, dict):
                sentiment = result.get('sentiment', ['Neutral'])[0] if isinstance(result.get('sentiment'), list) else result.get('sentiment', 'Neutral')
            elif isinstance(result, list) and len(result) > 0:
                sentiment = result[0].get('sentiment', 'Neutral') if isinstance(result[0], dict) else str(result[0])
            else:
                sentiment = str(result)
            
            stance = map_sentiment_to_stance(sentiment)
            
        except Exception as e:
            stance = 'Neutral'
        
        predictions.append({
            'tweet': tweet,
            'keyword': keyword,
            'original_stance': row['stance'],
            'pyabsa_prediction': stance
        })
    
    return pd.DataFrame(predictions)


def run_trained_model_inference(test_df: pd.DataFrame) -> pd.DataFrame:
    """Run inference using the trained PyABSA model."""
    print("\n" + "=" * 60)
    print("Running PyABSA Trained Model Inference")
    print("=" * 60)
    
    try:
        from pyabsa import AspectPolarityClassification as APC
        
        # Find the trained checkpoint
        checkpoint_path = None
        for item in MODEL_DIR.iterdir():
            if item.is_dir() and 'checkpoint' in item.name.lower():
                checkpoint_path = item
                break
        
        if checkpoint_path is None:
            # Try to find any model files
            for item in MODEL_DIR.iterdir():
                if item.suffix in ['.state_dict', '.config']:
                    checkpoint_path = MODEL_DIR
                    break
        
        if checkpoint_path:
            print(f"Loading trained model from: {checkpoint_path}")
            classifier = APC.SentimentClassifier(checkpoint=str(checkpoint_path), auto_device=True)
            model_loaded = True
        else:
            print("No trained checkpoint found, using pre-trained model")
            model_loaded = False
            
    except Exception as e:
        print(f"Could not load trained model: {e}")
        model_loaded = False
    
    if not model_loaded:
        return run_pretrained_inference(test_df)
    
    predictions = []
    
    print(f"\nProcessing {len(test_df)} samples...")
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Classifying"):
        tweet = str(row['tweet'])
        keyword = str(row['keyword'])
        
        try:
            # PyABSA format
            formatted_input = f"{tweet} $T$ {keyword}"
            result = classifier.predict(formatted_input)
            
            if hasattr(result, 'sentiment'):
                sentiment = result.sentiment
            elif isinstance(result, dict):
                sentiment = result.get('sentiment', 'Neutral')
            else:
                sentiment = str(result)
            
            stance = map_sentiment_to_stance(sentiment)
            
        except Exception:
            stance = 'Neutral'
        
        predictions.append({
            'tweet': tweet,
            'keyword': keyword,
            'original_stance': row['stance'],
            'pyabsa_prediction': stance
        })
    
    return pd.DataFrame(predictions)


def main():
    """Main PyABSA pipeline with training option."""
    parser = argparse.ArgumentParser(description='PyABSA ABSA with training')
    parser.add_argument('--train', action='store_true',
                       help='Train PyABSA model on dataset (default: use pre-trained)')
    parser.add_argument('--skip-if-exists', action='store_true',
                       help='Skip training if model already exists')
    args = parser.parse_args()
    
    print("=" * 60)
    print("PyABSA Aspect-Based Sentiment Analysis")
    print("=" * 60)
    
    # Create directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load processed data
    train_path = PROCESSED_DIR / "train_processed.csv"
    test_path = PROCESSED_DIR / "test_processed.csv"
    
    if not test_path.exists():
        print(f"Error: Processed test data not found at {test_path}")
        print("Please run data_preprocessor.py first!")
        sys.exit(1)
    
    print(f"\nLoading test data from: {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"  Test samples: {len(test_df)}")
    
    if args.train:
        if not train_path.exists():
            print(f"Error: Processed train data not found at {train_path}")
            sys.exit(1)
        
        # Check if model already exists
        if args.skip_if_exists and MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
            print(f"\nTrained model already exists at {MODEL_DIR}")
            print("Using existing model for inference...")
        else:
            print(f"\nLoading train data from: {train_path}")
            train_df = pd.read_csv(train_path)
            print(f"  Train samples: {len(train_df)}")
            
            # Split for validation (10%)
            from sklearn.model_selection import train_test_split
            train_data, val_data = train_test_split(
                train_df, test_size=0.1, random_state=42, stratify=train_df['stance']
            )
            
            # Try to train
            trainer = train_pyabsa_model(train_data, val_data)
        
        # Run inference with trained model
        predictions_df = run_trained_model_inference(test_df)
    else:
        # Use pre-trained/fallback
        predictions_df = run_pretrained_inference(test_df)
    
    # Save predictions
    output_path = RESULTS_DIR / "pyabsa_predictions.csv"
    predictions_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved predictions to: {output_path}")
    
    # Quick accuracy check
    correct = (predictions_df['original_stance'] == predictions_df['pyabsa_prediction']).sum()
    total = len(predictions_df)
    accuracy = correct / total * 100
    
    print("\n" + "=" * 60)
    print("PyABSA Quick Results")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    print("\nGround truth distribution:")
    print(predictions_df['original_stance'].value_counts().to_string())
    
    print("\nPrediction distribution:")
    print(predictions_df['pyabsa_prediction'].value_counts().to_string())
    
    print("\n✓ PyABSA inference complete!")
    
    return predictions_df


if __name__ == "__main__":
    main()
