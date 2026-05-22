#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-x_translate}"
INPUT_CSV="${1:-$BASE_DIR/Data/retweet_data_combined_labeled_by_retweet.csv}"
TEXT_COL="${TEXT_COL:-tweet}"
RUN_TAG="${RUN_TAG:-full_en_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$BASE_DIR/Data/$RUN_TAG}"
ROUTED_DIR="$RUN_DIR/routed"
TRANSLATED_DIR="$RUN_DIR/translated"
LOG_DIR="$RUN_DIR/logs"
MODEL_DIR="${MODEL_DIR:-$BASE_DIR/models/indiclid}"

ROUTE_GPU="${ROUTE_GPU:-0}"
NATIVE_GPU="${NATIVE_GPU:-0}"
ROMAN_GPU="${ROMAN_GPU:-1}"

ROUTE_CHUNK_SIZE="${ROUTE_CHUNK_SIZE:-100000}"
ROUTE_LID_BATCH_SIZE="${ROUTE_LID_BATCH_SIZE:-256}"
TRANSLATE_CHUNK_SIZE="${TRANSLATE_CHUNK_SIZE:-20000}"
TRANSLATE_BATCH_SIZE="${TRANSLATE_BATCH_SIZE:-64}"
INSPECT_QUANTILE="${INSPECT_QUANTILE:-0.10}"
HIGHRISK_QUANTILE="${HIGHRISK_QUANTILE:-0.05}"

FINAL_OUTPUT="$RUN_DIR/retweet_data_combined_labeled_by_retweet_en.csv"
NATIVE_OUTPUT="$TRANSLATED_DIR/hindi_script_en.csv"
ROMAN_OUTPUT="$TRANSLATED_DIR/hinglish_roman_en.csv"

mkdir -p "$ROUTED_DIR" "$TRANSLATED_DIR" "$LOG_DIR" "$MODEL_DIR"

if [[ ! -f "$INPUT_CSV" ]]; then
  echo "Input CSV not found: $INPUT_CSV" >&2
  exit 1
fi

echo "run_dir=$RUN_DIR"
echo "input_csv=$INPUT_CSV"
echo "text_col=$TEXT_COL"
echo "env_name=$ENV_NAME"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$ROUTE_GPU" micromamba run -n "$ENV_NAME" \
  python "$BASE_DIR/prepare_translation_routes.py" \
  --input-csv "$INPUT_CSV" \
  --text-col "$TEXT_COL" \
  --output-dir "$ROUTED_DIR" \
  --chunk-size "$ROUTE_CHUNK_SIZE" \
  --lid-batch-size "$ROUTE_LID_BATCH_SIZE" \
  --model-dir "$MODEL_DIR" \
  > "$LOG_DIR/01_route.log" 2>&1

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$NATIVE_GPU" micromamba run -n "$ENV_NAME" \
  python "$BASE_DIR/full_run_hindi_pipeline.py" \
  --input "$ROUTED_DIR/hindi_script_tweets.csv" \
  --text-col "$TEXT_COL" \
  --src-lang-col nllb_src_lang \
  --engine nllb \
  --out-translated "$NATIVE_OUTPUT" \
  --lowcos-out "$TRANSLATED_DIR/hindi_script_lowcos.csv" \
  --chunk-size "$TRANSLATE_CHUNK_SIZE" \
  --batch-size "$TRANSLATE_BATCH_SIZE" \
  --inspect-quantile "$INSPECT_QUANTILE" \
  --highrisk-quantile "$HIGHRISK_QUANTILE" \
  > "$LOG_DIR/02_translate_native.log" 2>&1 &
native_pid=$!

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$ROMAN_GPU" micromamba run -n "$ENV_NAME" \
  python "$BASE_DIR/full_run_hindi_pipeline.py" \
  --input "$ROUTED_DIR/hinglish_roman_tweets.csv" \
  --text-col "$TEXT_COL" \
  --src-lang-col nllb_src_lang \
  --engine nllb \
  --out-translated "$ROMAN_OUTPUT" \
  --lowcos-out "$TRANSLATED_DIR/hinglish_roman_lowcos.csv" \
  --chunk-size "$TRANSLATE_CHUNK_SIZE" \
  --batch-size "$TRANSLATE_BATCH_SIZE" \
  --inspect-quantile "$INSPECT_QUANTILE" \
  --highrisk-quantile "$HIGHRISK_QUANTILE" \
  > "$LOG_DIR/03_translate_roman.log" 2>&1 &
roman_pid=$!

wait "$native_pid"
wait "$roman_pid"

PYTHONUNBUFFERED=1 micromamba run -n "$ENV_NAME" \
  python "$BASE_DIR/merge_translation_outputs.py" \
  --english "$ROUTED_DIR/english_tweets.csv" \
  --native-translated "$NATIVE_OUTPUT" \
  --roman-translated "$ROMAN_OUTPUT" \
  --output "$FINAL_OUTPUT" \
  --text-col "$TEXT_COL" \
  > "$LOG_DIR/04_merge.log" 2>&1

echo "final_output=$FINAL_OUTPUT"
