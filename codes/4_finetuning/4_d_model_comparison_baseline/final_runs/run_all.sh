#!/bin/bash
# =============================================================================
# Final Model Comparison Pipeline - run_all.sh
# Runs all 6 stance detection models with detailed logging
# =============================================================================

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Paths
TRAIN_CSV="/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/4_finetuning/4_a_DataProcessing/data_formatting/master_train.csv"
TEST_CSV="/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/4_finetuning/4_a_DataProcessing/data_formatting/master_test.csv"
SHOTS_DIR="/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/4_finetuning/4_a_DataProcessing/data_formatting/jsons"
SHOTS_JSON="/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/4_finetuning/4_a_DataProcessing/data_formatting/jsons/kyra_modi_stance.json"
MISTRAL_MODEL="/scratch/ziv_baretto/Research_X/models/Mistral-7B-Instruct-v0.3"
LORA_ADAPTER="/scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/lora_adaptor/out_train_final/mistral7b_stance_lora"

RESULTS_DIR="$SCRIPT_DIR/results"
LOGS_DIR="$SCRIPT_DIR/logs"
SCRIPTS_DIR="$SCRIPT_DIR/scripts"

# Create directories
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

echo "=============================================="
echo "Final Model Comparison Pipeline"
echo "=============================================="
echo "Working directory: $SCRIPT_DIR"
echo "Started at: $(date)"
echo ""
echo "Train CSV: $TRAIN_CSV"
echo "Test CSV: $TEST_CSV"
echo ""

# Activate environment
echo "Activating partisan_env..."
eval "$(micromamba shell hook --shell bash)"
micromamba activate partisan_env

# =============================================================================
# Model 1: BERT
# =============================================================================
echo ""
echo "=============================================="
echo "Model 1/5: BERT"
echo "=============================================="
python "$SCRIPTS_DIR/bert_stance.py" \
    --train-csv "$TRAIN_CSV" \
    --test-csv "$TEST_CSV" \
    --output-dir "$RESULTS_DIR" \
    --log-file "$LOGS_DIR/bert.log" \
    --epochs 3 \
    2>&1 | tee -a "$LOGS_DIR/bert.log"


# =============================================================================
# Model 2: RoBERTa
# =============================================================================
echo ""
echo "=============================================="
echo "Model 2/5: RoBERTa"
echo "=============================================="
python "$SCRIPTS_DIR/roberta_stance.py" \
    --train-csv "$TRAIN_CSV" \
    --test-csv "$TEST_CSV" \
    --output-dir "$RESULTS_DIR" \
    --log-file "$LOGS_DIR/roberta.log" \
    --epochs 3 \
    2>&1 | tee -a "$LOGS_DIR/roberta.log"

# =============================================================================
# Model 3: Mistral Base (Zero-shot)
# =============================================================================
echo ""
echo "=============================================="
echo "Model 3/5: Mistral Base (Zero-shot)"
echo "=============================================="
python "$SCRIPTS_DIR/mistral_base_inference.py" \
    --model-dir "$MISTRAL_MODEL" \
    --test-csv "$TEST_CSV" \
    --output-dir "$RESULTS_DIR" \
    --few-shot 0 \
    --output-prefix "mistral_base" \
    2>&1 | tee "$LOGS_DIR/mistral_base.log"

# =============================================================================
# Model 4: Mistral Base (Few-shot)
# =============================================================================
echo ""
echo "=============================================="
echo "Model 4/5: Mistral Base (Few-shot)"
echo "=============================================="
python "$SCRIPTS_DIR/finetune_stance.py" \
    --input_csv "$TEST_CSV" \
    --model "$MISTRAL_MODEL" \
    --shots_dir "$SHOTS_DIR" \
    --shots_prefix "kyra" \
    --shots_json "$SHOTS_JSON" \
    --output_csv "$RESULTS_DIR/mistral_fewshot_predictions.csv" \
    --batch_size 16 \
    --log_file "$LOGS_DIR/mistral_fewshot.log" \
    2>&1 | tee "$LOGS_DIR/mistral_fewshot.log"

# =============================================================================
# Model 5: Mistral LoRA Fine-tuned (Few-shot)
# =============================================================================
echo ""
echo "=============================================="
echo "Model 5/5: Mistral LoRA Fine-tuned (Few-shot)"
echo "=============================================="
python "$SCRIPTS_DIR/finetune_stance.py" \
    --input_csv "$TEST_CSV" \
    --model "$MISTRAL_MODEL" \
    --lora_adapter "$LORA_ADAPTER" \
    --shots_dir "$SHOTS_DIR" \
    --shots_prefix "kyra" \
    --shots_json "$SHOTS_JSON" \
    --output_csv "$RESULTS_DIR/mistral_lora_predictions.csv" \
    --batch_size 16 \
    --log_file "$LOGS_DIR/mistral_lora.log" \
    2>&1 | tee "$LOGS_DIR/mistral_lora.log"

# =============================================================================
# Final Evaluation
# =============================================================================
echo ""
echo "=============================================="
echo "Running Final Evaluation"
echo "=============================================="
python "$SCRIPTS_DIR/final_evaluation.py" \
    --results-dir "$RESULTS_DIR" \
    --log-file "$LOGS_DIR/evaluation.log" \
    2>&1 | tee "$LOGS_DIR/evaluation.log"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo "Finished at: $(date)"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo "Logs saved to: $LOGS_DIR"
echo ""
echo "Output files:"
ls -la "$RESULTS_DIR"
echo ""
echo "Log files:"
ls -la "$LOGS_DIR"
