#!/usr/bin/env python3
"""
Download Mistral-7B-Instruct-v0.3 model from Hugging Face.

Usage:
    python download_mistral.py [--output_dir /path/to/save]

Requirements:
    - huggingface_hub (already installed)
    - You may need to login first: huggingface-cli login
"""

import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def main():
    parser = argparse.ArgumentParser(description="Download Mistral-7B-Instruct-v0.3")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/Mistral-7B-Instruct-v0.3",
        help="Directory to save the model (default: ./models/Mistral-7B-Instruct-v0.3)"
    )
    args = parser.parse_args()

    output_path = Path(args.output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {MODEL_ID} to {output_path}...")
    print("This may take a while depending on your internet speed (~15GB).\n")

    try:
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(output_path),
            local_dir_use_symlinks=False,  # Download actual files, not symlinks
            resume_download=True,          # Resume if interrupted
        )
        print(f"\n✓ Model downloaded successfully to: {output_path}")
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("\nIf you see an authentication error, run:")
        print("    huggingface-cli login")
        print("and enter your Hugging Face token (requires accepting model license).")
        raise

if __name__ == "__main__":
    main()
