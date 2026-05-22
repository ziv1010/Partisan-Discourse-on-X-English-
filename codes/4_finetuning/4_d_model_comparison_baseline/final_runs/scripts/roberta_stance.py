#!/usr/bin/env python3
"""
RoBERTa-based Stance Classification for Multi-Model Comparison.
Fine-tunes RoBERTa on stance detection task with aspect/keyword awareness.
"""

import os
import random
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


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_stance(stance: str):
    if pd.isna(stance):
        return None
    s = str(stance).strip().lower()
    if not s:
        return None
    s = s.replace('favour', 'for').replace('favor', 'for').replace('nuetral', 'neutral')
    if s in ['for', 'positive']:
        return 'For'
    if s in ['against', 'negative']:
        return 'Against'
    if s in ['neutral', 'unrelated', 'irrelevant']:
        return 'Neutral'
    return None


def normalize_frame(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    before = len(df)
    df = df.copy()
    df['stance'] = df['stance'].apply(normalize_stance)
    df = df[df['stance'].isin(LABEL2ID)].reset_index(drop=True)
    dropped = before - len(df)
    print(f"\n  {split_name} samples: {len(df)}")
    if dropped:
        print(f"  Dropped {dropped} {split_name} rows with blank/unsupported stance labels")
    print(f"  {split_name} stance distribution:")
    print(df['stance'].value_counts().to_string())
    return df


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


def load_data(train_path: Path, test_path: Path, dev_path: Path = None):
    """Load and preprocess training and test data."""
    print(f"\nLoading data from:")
    print(f"  Train: {train_path}")
    if dev_path:
        print(f"  Dev: {dev_path}")
    print(f"  Test: {test_path}")
    
    train_df = normalize_frame(pd.read_csv(train_path), "Train")
    dev_df = normalize_frame(pd.read_csv(dev_path), "Dev") if dev_path else None
    test_df = normalize_frame(pd.read_csv(test_path), "Test")
    
    return train_df, dev_df, test_df


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
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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
    parser.add_argument('--train-csv', type=str, default=None, help='Path to train CSV')
    parser.add_argument('--dev-csv', type=str, default=None, help='Optional fixed dev CSV')
    parser.add_argument('--test-csv', type=str, default=None, help='Path to test CSV')
    parser.add_argument('--output-dir', type=str, default=str(RESULTS_DIR), help='Output directory')
    parser.add_argument('--model-dir', type=str, default=str(MODEL_DIR), help='Model/checkpoint directory')
    parser.add_argument('--prediction-file', type=str, default='roberta_predictions.csv', help='Prediction CSV filename')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max-length', type=int, default=256, help='Max sequence length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for initialization and dataloader shuffling')
    parser.add_argument('--split-seed', type=int, default=42, help='Used only when --dev-csv is not supplied')
    parser.add_argument('--skip-train', action='store_true', help='Skip training, load existing model')
    args = parser.parse_args()
    set_all_seeds(args.seed)
    
    print("=" * 60)
    print("RoBERTa Stance Classification")
    print("=" * 60)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Create directories
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    train_path = Path(args.train_csv) if args.train_csv else DATA_DIR / "master_train.csv"
    dev_path = Path(args.dev_csv) if args.dev_csv else None
    test_path = Path(args.test_csv) if args.test_csv else DATA_DIR / "master_test.csv"
    train_df, dev_df, test_df = load_data(train_path, test_path, dev_path)
    
    # Initialize tokenizer
    print("\nLoading RoBERTa tokenizer and model...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Prepare labels
    train_labels = [LABEL2ID[s] for s in train_df['stance']]
    test_labels = [LABEL2ID[s] for s in test_df['stance']]
    
    # Create datasets. Use the fixed dev set when supplied; otherwise keep the
    # legacy fixed split behavior for backwards compatibility.
    if dev_df is not None:
        train_data = train_df
        val_data = dev_df
    else:
        val_size = int(0.1 * len(train_df))
        val_indices = np.random.RandomState(args.split_seed).choice(len(train_df), val_size, replace=False)
        train_indices = [i for i in range(len(train_df)) if i not in val_indices]
        train_data = train_df.iloc[train_indices]
        val_data = train_df.iloc[val_indices]
    
    train_dataset = StanceDataset(
        train_data['tweet'].tolist(),
        train_data['keyword'].tolist(),
        [LABEL2ID[s] for s in train_data['stance']],
        tokenizer,
        args.max_length
    )
    
    val_dataset = StanceDataset(
        val_data['tweet'].tolist(),
        val_data['keyword'].tolist(),
        [LABEL2ID[s] for s in val_data['stance']],
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
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    print(f"\n  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Initialize or load model
    model_path = model_dir / "roberta_stance_model.pt"
    
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
    
    output_path = output_dir / args.prediction_file
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
