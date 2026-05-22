#!/usr/bin/env python3
"""
BERT Stance Classification for Final Runs  
Trains BERT on master_train.csv, evaluates on master_test.csv.
"""

import os
import random
import sys
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm

# Paths
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
LOGS_DIR = SCRIPT_DIR.parent / "logs"
MODEL_DIR = SCRIPT_DIR.parent / "models" / "bert"

LABEL2ID = {"For": 0, "Against": 1, "Neutral": 2}
ID2LABEL = {0: "For", 1: "Against", 2: "Neutral"}


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


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_stance(stance: str) -> str:
    """Normalize stance labels."""
    if pd.isna(stance):
        return None
    s = str(stance).lower().strip()
    if not s:
        return None
    s = s.replace('favour', 'for').replace('favor', 'for').replace('nuetral', 'neutral')
    if s in ['for', 'positive']:
        return 'For'
    elif s in ['against', 'negative']:
        return 'Against'
    elif s in ['neutral', 'unrelated', 'irrelevant']:
        return 'Neutral'
    return None


def load_three_class_csv(path: str, logger, split_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    before = len(df)
    df['stance'] = df['stance'].apply(normalize_stance)
    df = df[df['stance'].isin(LABEL2ID)].reset_index(drop=True)
    dropped = before - len(df)
    logger.info(f"  {split_name} samples: {len(df)}, Unique tweets: {df['tweet'].nunique()}")
    if dropped:
        logger.info(f"  Dropped {dropped} {split_name} rows with blank/unsupported stance labels")
    logger.info(f"  {split_name} stance distribution:\n{df['stance'].value_counts().to_string()}")
    return df


class StanceDataset(Dataset):
    """Dataset for stance classification with aspect-aware input."""
    
    def __init__(self, texts, keywords, labels, tokenizer, max_length=256):
        self.texts = texts
        self.keywords = keywords
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        keyword = str(self.keywords[idx])
        
        # Format: [CLS] keyword [SEP] tweet [SEP]
        combined = f"{keyword} [SEP] {text}"
        
        encoding = self.tokenizer(
            combined,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def train_model(model, train_loader, val_loader, device, model_dir: Path, epochs=3, lr=2e-5, logger=None):
    """Train BERT model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val_acc = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        val_acc = correct / total
        if logger:
            logger.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_dir / "best_model.pt")
    
    return model


def predict(model, test_loader, device):
    """Run inference on test set."""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
    
    return [ID2LABEL[p] for p in predictions]


def main():
    parser = argparse.ArgumentParser(description='BERT Stance Classification')
    parser.add_argument('--train-csv', type=str, required=True, help='Path to train CSV')
    parser.add_argument('--dev-csv', type=str, default=None, help='Optional fixed dev CSV')
    parser.add_argument('--test-csv', type=str, required=True, help='Path to test CSV')
    parser.add_argument('--output-dir', type=str, default=str(RESULTS_DIR), help='Output directory')
    parser.add_argument('--model-dir', type=str, default=str(MODEL_DIR), help='Model/checkpoint directory')
    parser.add_argument('--log-file', type=str, default=str(LOGS_DIR / 'bert.log'), help='Log file path')
    parser.add_argument('--prediction-file', type=str, default='bert_predictions.csv', help='Prediction CSV filename')
    parser.add_argument('--epochs', type=int, default=3, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for initialization and dataloader shuffling')
    parser.add_argument('--split-seed', type=int, default=42, help='Used only when --dev-csv is not supplied')
    args = parser.parse_args()
    set_all_seeds(args.seed)
    
    # Setup logging
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_path)
    
    logger.info("=" * 60)
    logger.info("BERT Stance Classification")
    logger.info("=" * 60)
    
    # Create directories
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load tokenizer and model
    logger.info("Loading BERT model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    ).to(device)
    
    # Load data
    logger.info(f"Loading train data from: {args.train_csv}")
    train_df = load_three_class_csv(args.train_csv, logger, "Train")
    if args.dev_csv:
        logger.info(f"Loading fixed dev data from: {args.dev_csv}")
        val_data = load_three_class_csv(args.dev_csv, logger, "Dev")
        train_data = train_df
    else:
        train_data, val_data = train_test_split(
            train_df, test_size=0.1, random_state=args.split_seed, stratify=train_df['stance']
        )
    
    logger.info(f"Loading test data from: {args.test_csv}")
    test_df = load_three_class_csv(args.test_csv, logger, "Test")
    
    # Create datasets
    train_labels = [LABEL2ID[s] for s in train_data['stance']]
    val_labels = [LABEL2ID[s] for s in val_data['stance']]
    test_labels = [LABEL2ID[s] for s in test_df['stance']]
    
    train_dataset = StanceDataset(train_data['tweet'].tolist(), train_data['keyword'].tolist(), train_labels, tokenizer)
    val_dataset = StanceDataset(val_data['tweet'].tolist(), val_data['keyword'].tolist(), val_labels, tokenizer)
    test_dataset = StanceDataset(test_df['tweet'].tolist(), test_df['keyword'].tolist(), test_labels, tokenizer)
    
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Train
    logger.info(f"\nTraining for {args.epochs} epochs...")
    model = train_model(model, train_loader, val_loader, device, model_dir=model_dir, epochs=args.epochs, logger=logger)
    
    # Load best model
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
    
    # Predict
    logger.info("\nRunning inference on test set...")
    predictions = predict(model, test_loader, device)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'tweet': test_df['tweet'].tolist(),
        'keyword': test_df['keyword'].tolist(),
        'original_stance': test_df['stance'].tolist(),
        'bert_prediction': predictions
    })
    
    # Save predictions
    output_path = output_dir / args.prediction_file
    results_df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved predictions to: {output_path}")
    
    # Report accuracy
    correct = (results_df['original_stance'] == results_df['bert_prediction']).sum()
    total = len(results_df)
    accuracy = correct / total * 100
    macro_f1 = f1_score(results_df['original_stance'], results_df['bert_prediction'], average='macro', labels=list(LABEL2ID), zero_division=0)
    weighted_f1 = f1_score(results_df['original_stance'], results_df['bert_prediction'], average='weighted', labels=list(LABEL2ID), zero_division=0)
    
    logger.info(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
    logger.info(f"Macro F1: {macro_f1:.4f}")
    logger.info(f"Weighted F1: {weighted_f1:.4f}")
    logger.info(f"\nGround truth distribution:\n{results_df['original_stance'].value_counts().to_string()}")
    logger.info(f"\nPrediction distribution:\n{results_df['bert_prediction'].value_counts().to_string()}")
    logger.info("✓ BERT training and inference complete!")


if __name__ == "__main__":
    main()
