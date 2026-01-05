#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mistral Base Model Inference (Few-shot & Zero-shot)
Runs inference on test data using base Mistral-7B-Instruct without LoRA adapters.
Supports few-shot prompting using examples from JSON files.
"""

import os, argparse, json, re, random
from typing import Dict, Tuple, List, Optional
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Paths
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR.parent / "results"
JSON_EXAMPLES_DIR = BASE_DIR.parent.parent / "4_a_DataProcessing" / "data_formatting" / "jsons"

INSTR_HDR = "### Instruction:\n"
INPUT_HDR = "### Input:\n"
RESP_HDR  = "### Response:\n"

ALLOWED_STANCES = {"for": "For", "against": "Against", "neutral": "Neutral", "unrelated": "Unrelated"}


def normalize_stance(x: str) -> str:
    """Normalize stance labels to standard format."""
    if x is None:
        return "Neutral"
    s = str(x).strip().lower().replace(".", "")
    s = s.replace("favour", "for").replace("in_favor", "for").replace("in favour", "for")
    s = s.replace("favor", "for")
    s = s.replace("oppose", "against")
    s = s.replace("nuetral", "neutral")
    s = s.replace("positive", "for")
    s = s.replace("negative", "against")
    return ALLOWED_STANCES.get(s, "Neutral")


def load_fewshot_examples(keyword: str, fallback_keyword: str = "modi", n_shots: int = 3) -> List[Dict]:
    """Load few-shot examples from JSON files."""
    # Try specific keyword file first
    json_path = JSON_EXAMPLES_DIR / f"kyra_{keyword}_stance.json"
    
    if not json_path.exists():
        # normalize keyword for file lookup (e.g. "farm laws" -> "farm_laws")
        norm_keyword = keyword.replace(" ", "_").lower()
        json_path = JSON_EXAMPLES_DIR / f"kyra_{norm_keyword}_stance.json"
    
    # Fallback if not found
    if not json_path.exists():
        # print(f"Warning: No examples found for '{keyword}', using fallback '{fallback_keyword}'")
        json_path = JSON_EXAMPLES_DIR / f"kyra_{fallback_keyword}_stance.json"
    
    if not json_path.exists():
        return []

    try:
        with open(json_path, 'r') as f:
            examples = json.load(f)
        
        # Randomly select n_shots
        if len(examples) > n_shots:
            return random.sample(examples, n_shots)
        return examples
    except Exception as e:
        print(f"Error loading examples from {json_path}: {e}")
        return []


def build_prompt(tweet: str, keyword: str, examples: List[Dict] = []) -> str:
    """Build inference prompt with optional few-shot examples."""
    instruction = (
        'Given a tweet and a target keyword, classify the tweet\'s stance toward the target as one of '
        '"For", "Against", "Neutral", or "Unrelated" and provide a concise reason grounded in the tweet. '
        'Return ONLY a compact JSON object with keys "stance" and "reason" (no extra text).'
    )
    
    prompt = f"{INSTR_HDR}{instruction}\n\n"
    
    # Add few-shot examples
    for ex in examples:
        ex_tweet = ex.get("statement", "")
        ex_target = ex.get("entity", "")
        ex_stance = normalize_stance(ex.get("stance", ""))
        ex_reason = ex.get("reason", "No reason provided.")
        
        # Normalize stance for the example output to match expected output format
        # The JSONs have "positive"/"negative", we want "For"/"Against"
        
        model_input = f"target: {ex_target}\n\ntweet: {ex_tweet}"
        gold_json = json.dumps({"stance": ex_stance, "reason": ex_reason}, ensure_ascii=False)
        
        prompt += f"{INPUT_HDR}{model_input}\n\n{RESP_HDR}{gold_json}\n\n"

    # Add current sample
    model_input = f"target: {keyword}\n\ntweet: {tweet}"
    prompt += f"{INPUT_HDR}{model_input}\n\n{RESP_HDR}"
    
    return prompt


def parse_model_output(output: str) -> str:
    """Parse model output to extract stance."""
    # Try to find JSON in output
    try:
        # Find JSON-like pattern
        json_match = re.search(r'\{[^}]+\}', output)
        if json_match:
            parsed = json.loads(json_match.group())
            stance = parsed.get("stance", "Neutral")
            return normalize_stance(stance)
    except (json.JSONDecodeError, AttributeError):
        pass
    
    # Fallback: look for stance keywords in output
    output_lower = output.lower()
    if "against" in output_lower:
        return "Against"
    elif "for" in output_lower or "favor" in output_lower or "favour" in output_lower:
        return "For"
    elif "neutral" in output_lower:
        return "Neutral"
    
    return "Neutral"


def load_model_and_tokenizer(model_dir: str, device_map: str = "auto"):
    """Load base Mistral model without any adapters."""
    print(f"Loading tokenizer from: {model_dir}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # For generation
    
    print(f"Loading model from: {model_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    
    return model, tokenizer


def run_inference(model, tokenizer, test_df: pd.DataFrame, max_new_tokens: int = 128, few_shot: int = 0):
    """Run inference on test data."""
    predictions = []
    device = next(model.parameters()).device
    
    print(f"\nRunning inference on {len(test_df)} samples (Few-shot: {few_shot})...")
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Inferring"):
        tweet = str(row['tweet'])
        keyword = str(row['keyword'])
        original_stance = row.get('stance', row.get('original_stance', 'Unknown'))
        
        # Get examples if few-shot enabled
        examples = []
        if few_shot > 0:
            examples = load_fewshot_examples(keyword, n_shots=few_shot)
        
        # Build prompt
        prompt = build_prompt(tweet, keyword, examples)
        
        # Tokenize (truncate from left to keep the end instructions/examples if prompt is too long)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048) # Increased context for few-shot
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode only the generated part
        generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Parse stance from output
        predicted_stance = parse_model_output(generated)
        
        predictions.append({
            'tweet': tweet,
            'keyword': keyword,
            'original_stance': normalize_stance(str(original_stance)),
            'mistral_prediction': predicted_stance,
            'raw_output': generated[:200]
        })
    
    return pd.DataFrame(predictions)


def main():
    parser = argparse.ArgumentParser(description='Run base Mistral inference on test data')
    parser.add_argument('--model-dir', type=str, 
                       default='/scratch/ziv_baretto/Research_X/models/Mistral-7B-Instruct-v0.3',
                       help='Path to Mistral model directory')
    parser.add_argument('--test-csv', type=str,
                       default=str(BASE_DIR.parent / 'processed_data' / 'test_processed.csv'),
                       help='Path to test CSV file')
    parser.add_argument('--output-dir', type=str,
                       default=str(RESULTS_DIR),
                       help='Directory to save results')
    parser.add_argument('--few-shot', type=int, default=0,
                       help='Number of few-shot examples (0 for zero-shot)')
    parser.add_argument('--output-prefix', type=str, default='mistral_base',
                       help='Prefix for output filename')
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Mistral Inference (Few-shot: {args.few_shot})")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    test_path = Path(args.test_csv)
    if not test_path.exists():
        test_path = BASE_DIR.parent / 'processed_data' / 'test_processed.csv'
    
    if not test_path.exists():
        print(f"Error: Test data not found at {test_path}")
        return
    
    print(f"\nLoading test data from: {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"  Samples: {len(test_df)}")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_dir)
    
    # Run inference
    predictions_df = run_inference(model, tokenizer, test_df, few_shot=args.few_shot)
    
    # Save predictions
    output_filename = f"{args.output_prefix}_predictions.csv"
    output_path = output_dir / output_filename
    predictions_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved predictions to: {output_path}")
    
    # Quick accuracy check
    correct = (predictions_df['original_stance'] == predictions_df['mistral_prediction']).sum()
    total = len(predictions_df)
    accuracy = correct / total * 100
    
    print("\n" + "=" * 60)
    print("Mistral Inference Results")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    print("\nGround truth distribution:")
    print(predictions_df['original_stance'].value_counts().to_string())
    
    print("\nPrediction distribution:")
    print(predictions_df['mistral_prediction'].value_counts().to_string())
    
    return predictions_df


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
