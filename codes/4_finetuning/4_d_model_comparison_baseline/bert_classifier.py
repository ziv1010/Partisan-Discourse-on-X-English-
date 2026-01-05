"""
BERT-based Stance Classifier
Fine-tunes BERT for 3-class stance classification using native PyTorch training.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from transformers import BertTokenizer, BertForSequenceClassification

# Paths
BASE_DIR = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / "processed_data"
RESULTS_DIR = BASE_DIR / "results"
MODEL_DIR = BASE_DIR / "bert_model"

# Label mapping
LABEL2ID = {"For": 0, "Against": 1, "Neutral": 2}
ID2LABEL = {0: "For", 1: "Against", 2: "Neutral"}


class StanceDataset(Dataset):
    """Custom Dataset for stance classification with aspect-aware input."""
    
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
        encoding = self.tokenizer(
            keyword,
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(LABEL2ID[self.labels[idx]])
        
        return item


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate the model."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy, f1


def train_bert_classifier(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """Fine-tune BERT for stance classification using native PyTorch."""
    print("\n" + "=" * 60)
    print("Training BERT Stance Classifier")
    print("=" * 60)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load tokenizer and model
    model_name = "bert-base-uncased"
    print(f"\nLoading {model_name}...")
    
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    model.to(device)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = StanceDataset(
        texts=train_df['tweet'].tolist(),
        keywords=train_df['keyword'].tolist(),
        labels=train_df['stance'].tolist(),
        tokenizer=tokenizer
    )
    
    val_dataset = StanceDataset(
        texts=val_df['tweet'].tolist(),
        keywords=val_df['keyword'].tolist(),
        labels=val_df['stance'].tolist(),
        tokenizer=tokenizer
    )
    
    # Dataloaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2)
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    # Optimizer and scheduler
    num_epochs = 5
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    scheduler = LinearLR(
        optimizer, 
        start_factor=1.0, 
        end_factor=0.0, 
        total_iters=total_steps
    )
    
    # Training loop
    print("\nStarting training...")
    best_f1 = 0
    patience = 3
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        
        # Evaluate
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(MODEL_DIR))
            tokenizer.save_pretrained(str(MODEL_DIR))
            print(f"  ✓ Saved best model (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping triggered (no improvement for {patience} epochs)")
                break
    
    print(f"\nBest validation F1: {best_f1:.4f}")
    print(f"Model saved to: {MODEL_DIR}")
    
    return model, tokenizer


def predict_with_bert(test_df: pd.DataFrame, model_dir: Path):
    """Run inference on test set with trained BERT model."""
    print("\n" + "=" * 60)
    print("Running BERT Inference on Test Set")
    print("=" * 60)
    
    # Check if model exists
    if not (model_dir / "config.json").exists():
        print(f"Error: No trained model found at {model_dir}")
        return None
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model from: {model_dir}")
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    predictions = []
    
    print(f"\nPredicting on {len(test_df)} samples...")
    with torch.no_grad():
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
            text = str(row['tweet'])
            keyword = str(row['keyword'])
            
            # Tokenize
            encoding = tokenizer(
                keyword,
                text,
                truncation=True,
                max_length=256,
                padding='max_length',
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # Predict
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred_id = torch.argmax(outputs.logits, dim=1).item()
            pred_label = ID2LABEL[pred_id]
            
            predictions.append({
                'tweet': text,
                'keyword': keyword,
                'original_stance': row['stance'],
                'bert_prediction': pred_label
            })
    
    return pd.DataFrame(predictions)


def main():
    """Main BERT training and inference pipeline."""
    print("=" * 60)
    print("BERT Stance Classification")
    print("=" * 60)
    
    # Check for GPU
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available, using CPU (training will be slower)")
    
    # Create directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load processed data
    train_path = PROCESSED_DIR / "train_processed.csv"
    test_path = PROCESSED_DIR / "test_processed.csv"
    
    if not train_path.exists() or not test_path.exists():
        print(f"Error: Processed data not found!")
        print("Please run data_preprocessor.py first!")
        sys.exit(1)
    
    print(f"\nLoading train data from: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"  Samples: {len(train_df)}")
    
    print(f"\nLoading test data from: {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"  Samples: {len(test_df)}")
    
    # Split train into train/val (90/10)
    train_data, val_data = train_test_split(
        train_df, 
        test_size=0.1, 
        random_state=42,
        stratify=train_df['stance']
    )
    print(f"\nTrain/Val split: {len(train_data)}/{len(val_data)}")
    
    # Check if model already exists
    if (MODEL_DIR / "config.json").exists():
        print(f"\nFound existing model at {MODEL_DIR}")
        print("Skipping training, using existing model for inference...")
    else:
        # Train model
        train_bert_classifier(train_data, val_data)
    
    # Run inference on test set
    predictions_df = predict_with_bert(test_df, MODEL_DIR)
    
    if predictions_df is None:
        print("✗ BERT inference failed!")
        sys.exit(1)
    
    # Save predictions
    output_path = RESULTS_DIR / "bert_predictions.csv"
    predictions_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved predictions to: {output_path}")
    
    # Quick accuracy check
    correct = (predictions_df['original_stance'] == predictions_df['bert_prediction']).sum()
    total = len(predictions_df)
    accuracy = correct / total * 100
    
    print("\n" + "=" * 60)
    print("BERT Quick Results")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    print("\nGround truth distribution:")
    print(predictions_df['original_stance'].value_counts().to_string())
    
    print("\nPrediction distribution:")
    print(predictions_df['bert_prediction'].value_counts().to_string())
    
    print("\n✓ BERT training and inference complete!")
    
    return predictions_df


if __name__ == "__main__":
    main()
