#!/usr/bin/env python3
"""
RoBERTa-based Stance Classification for Multi-Model Comparison.
Fine-tunes RoBERTa on stance detection task with aspect/keyword awareness.
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
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "4_a_DataProcessing" / "data_formatting"
RESULTS_DIR = BASE_DIR / "results"
MODEL_DIR = BASE_DIR / "roberta_model"

# Label mapping
LABEL2ID = {'For': 0, 'Against': 1, 'Neutral': 2}
ID2LABEL = {0: 'For', 1: 'Against', 2: 'Neutral'}


class StanceDataset(Dataset):
    """Dataset for stance classification with aspect awareness."""
    
    def __init__(self, texts, aspects, labels, tokenizer, max_length=256):
        self.texts = texts
        self.aspects = aspects
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        aspect = str(self.aspects[idx])
        
        # Combine text and aspect for aspect-aware classification
        # Format: "[CLS] aspect [SEP] text [SEP]"
        combined = f"Topic: {aspect}. Tweet: {text}"
        
        encoding = self.tokenizer(
            combined,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_data(train_path: Path, test_path: Path):
    """Load and preprocess training and test data."""
    print(f"\nLoading data from:")
    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Filter valid stance labels
    valid_stances = ['For', 'Against', 'Neutral']
    train_df = train_df[train_df['stance'].isin(valid_stances)].reset_index(drop=True)
    test_df = test_df[test_df['stance'].isin(valid_stances)].reset_index(drop=True)
    
    print(f"\n  Train samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")
    
    print(f"\n  Train stance distribution:")
    print(train_df['stance'].value_counts().to_string())
    
    return train_df, test_df


def train_model(model, train_loader, val_loader, device, epochs=3, lr=2e-5):
    """Train the RoBERTa model."""
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    best_val_f1 = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        
        print(f"\n  Epoch {epoch+1}: Train Acc={train_acc:.4f}, Train F1={train_f1:.4f}, "
              f"Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            print(f"  ✓ New best model (F1={val_f1:.4f})")
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model


def evaluate_model(model, test_loader, device):
    """Evaluate the model on test set."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(outputs.logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def main():
    parser = argparse.ArgumentParser(description='RoBERTa Stance Classification')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max-length', type=int, default=256, help='Max sequence length')
    parser.add_argument('--skip-train', action='store_true', help='Skip training, load existing model')
    args = parser.parse_args()
    
    print("=" * 60)
    print("RoBERTa Stance Classification")
    print("=" * 60)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Create directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    train_path = DATA_DIR / "master_train.csv"
    test_path = DATA_DIR / "master_test.csv"
    train_df, test_df = load_data(train_path, test_path)
    
    # Initialize tokenizer
    print("\nLoading RoBERTa tokenizer and model...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Prepare labels
    train_labels = [LABEL2ID[s] for s in train_df['stance']]
    test_labels = [LABEL2ID[s] for s in test_df['stance']]
    
    # Create datasets
    # Use 10% of training for validation
    val_size = int(0.1 * len(train_df))
    val_indices = np.random.RandomState(42).choice(len(train_df), val_size, replace=False)
    train_indices = [i for i in range(len(train_df)) if i not in val_indices]
    
    train_dataset = StanceDataset(
        train_df.iloc[train_indices]['tweet'].tolist(),
        train_df.iloc[train_indices]['keyword'].tolist(),
        [train_labels[i] for i in train_indices],
        tokenizer,
        args.max_length
    )
    
    val_dataset = StanceDataset(
        train_df.iloc[val_indices]['tweet'].tolist(),
        train_df.iloc[val_indices]['keyword'].tolist(),
        [train_labels[i] for i in val_indices],
        tokenizer,
        args.max_length
    )
    
    test_dataset = StanceDataset(
        test_df['tweet'].tolist(),
        test_df['keyword'].tolist(),
        test_labels,
        tokenizer,
        args.max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    print(f"\n  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Initialize or load model
    model_path = MODEL_DIR / "roberta_stance_model.pt"
    
    if args.skip_train and model_path.exists():
        print(f"\nLoading existing model from {model_path}")
        model = RobertaForSequenceClassification.from_pretrained(
            'roberta-base',
            num_labels=3,
            id2label=ID2LABEL,
            label2id=LABEL2ID
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        model = RobertaForSequenceClassification.from_pretrained(
            'roberta-base',
            num_labels=3,
            id2label=ID2LABEL,
            label2id=LABEL2ID
        )
        
        model.to(device)
        
        print("\n" + "=" * 60)
        print("Training RoBERTa Model")
        print("=" * 60)
        
        model = train_model(
            model, train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr
        )
        
        # Save model
        torch.save(model.state_dict(), model_path)
        print(f"\n✓ Model saved to: {model_path}")
    
    model.to(device)
    
    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluating on Test Set")
    print("=" * 60)
    
    preds, labels, probs = evaluate_model(model, test_loader, device)
    
    # Convert to label names
    pred_labels = [ID2LABEL[p] for p in preds]
    true_labels = [ID2LABEL[l] for l in labels]
    
    # Print results
    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=['For', 'Against', 'Neutral']))
    
    # Save predictions
    results_df = pd.DataFrame({
        'tweet': test_df['tweet'].tolist(),
        'keyword': test_df['keyword'].tolist(),
        'original_stance': true_labels,
        'roberta_prediction': pred_labels,
        'prob_for': probs[:, 0],
        'prob_against': probs[:, 1],
        'prob_neutral': probs[:, 2]
    })
    
    output_path = RESULTS_DIR / "roberta_predictions.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved predictions to: {output_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("RoBERTa Stance Classification Results")
    print("=" * 60)
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Macro F1: {macro_f1*100:.2f}%")
    
    print("\nGround truth distribution:")
    print(pd.Series(true_labels).value_counts().to_string())
    
    print("\nPrediction distribution:")
    print(pd.Series(pred_labels).value_counts().to_string())
    
    print("\n✓ RoBERTa evaluation complete!")
    
    return results_df


if __name__ == "__main__":
    main()
