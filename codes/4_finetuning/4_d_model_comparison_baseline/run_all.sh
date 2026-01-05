#!/bin/bash
# Master runner script for ABSA baseline comparison
# Runs all scripts in sequence: preprocessing, PyABSA, BERT, Mistral Base, evaluation

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================"
echo "ABSA Baseline Comparison - Full Pipeline"
echo "========================================"
echo "Working directory: $SCRIPT_DIR"
echo "Started at: $(date)"
echo ""

# Activate environment
echo "Activating partisan_env..."
eval "$(micromamba shell hook --shell bash)"
micromamba activate partisan_env

# Parse arguments
CLEANING_LEVEL="${1:-basic}"  # default: basic
RUN_MISTRAL="${2:-yes}"       # default: yes

echo "Cleaning level: $CLEANING_LEVEL"
echo "Run Mistral base: $RUN_MISTRAL"
echo ""

echo "========================================"
echo "Step 1: Data Preprocessing"
echo "========================================"
python data_preprocessor.py --cleaning $CLEANING_LEVEL

echo ""
echo "========================================"
echo "Step 2: PyABSA Sentiment Analysis"
echo "========================================"
python pyabsa_absa.py

echo ""
echo "========================================"
echo "Step 3: BERT Stance Classification"
echo "========================================"
# Remove existing model to retrain with new cleaning
rm -rf bert_model/
python bert_classifier.py

if [ "$RUN_MISTRAL" = "yes" ]; then
    echo ""
    echo "========================================"
    echo "Step 4: Mistral Base Model Inference"
    echo "========================================"
    python mistral_baseline/mistral_base_inference.py \
        --model-dir /scratch/ziv_baretto/Research_X/models/Mistral-7B-Instruct-v0.3 \
        --test-csv processed_data/test_processed.csv \
        --output-dir results/
fi

echo ""
echo "========================================"
echo "Step 5: Evaluation & Comparison"
echo "========================================"
python run_evaluation.py

echo ""
echo "========================================"
echo "Pipeline Complete!"
echo "========================================"
echo "Finished at: $(date)"
echo ""
echo "Results saved to: $SCRIPT_DIR/results/"
ls -la results/
