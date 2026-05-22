#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${1:?usage: resume_native_translation_multigpu.sh /abs/path/to/run_dir}"
ENV_NAME="${ENV_NAME:-x_translate}"
TEXT_COL="${TEXT_COL:-tweet}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3}"
TRANSLATE_CHUNK_SIZE="${TRANSLATE_CHUNK_SIZE:-20000}"
TRANSLATE_BATCH_SIZE="${TRANSLATE_BATCH_SIZE:-96}"
DTYPE="${DTYPE:-bfloat16}"
INSPECT_QUANTILE="${INSPECT_QUANTILE:-0.10}"
HIGHRISK_QUANTILE="${HIGHRISK_QUANTILE:-0.05}"

IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_CSV"
NUM_SHARDS="${#GPU_IDS[@]}"
if [[ "$NUM_SHARDS" -lt 1 ]]; then
  echo "No GPU ids provided" >&2
  exit 1
fi

ROUTED_DIR="$RUN_DIR/routed"
TRANSLATED_DIR="$RUN_DIR/translated"
LOG_DIR="$RUN_DIR/logs"
WORK_DIR="$RUN_DIR/work/native_multigpu"
SHARD_INPUT_DIR="$WORK_DIR/inputs"
SHARD_OUTPUT_DIR="$WORK_DIR/outputs"
MANIFEST_PATH="$WORK_DIR/manifest.json"

mkdir -p "$SHARD_OUTPUT_DIR" "$LOG_DIR"

PYTHONUNBUFFERED=1 micromamba run -n "$ENV_NAME" \
  python "$BASE_DIR/prepare_native_multigpu_resume.py" \
  --run-dir "$RUN_DIR" \
  --num-shards "$NUM_SHARDS" \
  > "$LOG_DIR/05_prepare_native_multigpu.log" 2>&1

expected_total=$(micromamba run -n "$ENV_NAME" python - <<'PY' "$MANIFEST_PATH"
import json
import sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
print(manifest["native_total_rows"])
PY
)

shard_paths=()
for idx in $(seq 0 $((NUM_SHARDS - 1))); do
  shard_paths+=("$SHARD_INPUT_DIR/part_$(printf '%02d' "$idx").csv")
done

pids=()
for idx in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPU_IDS[$idx]}"
  input_csv="${shard_paths[$idx]}"
  output_csv="$SHARD_OUTPUT_DIR/shard_$(printf '%02d' "$idx").csv"
  lowcos_csv="$SHARD_OUTPUT_DIR/shard_$(printf '%02d' "$idx")_lowcos.csv"
  log_path="$LOG_DIR/05_native_multigpu_shard_$(printf '%02d' "$idx").log"

  PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES="$gpu" \
    micromamba run -n "$ENV_NAME" \
    python "$BASE_DIR/full_run_hindi_pipeline.py" \
    --input "$input_csv" \
    --text-col "$TEXT_COL" \
    --src-lang-col nllb_src_lang \
    --engine nllb \
    --out-translated "$output_csv" \
    --lowcos-out "$lowcos_csv" \
    --chunk-size "$TRANSLATE_CHUNK_SIZE" \
    --batch-size "$TRANSLATE_BATCH_SIZE" \
    --dtype "$DTYPE" \
    --inspect-quantile "$INSPECT_QUANTILE" \
    --highrisk-quantile "$HIGHRISK_QUANTILE" \
    > "$log_path" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

PYTHONUNBUFFERED=1 micromamba run -n "$ENV_NAME" \
  python "$BASE_DIR/assemble_native_multigpu_outputs.py" \
  --prefix "$TRANSLATED_DIR/hindi_script_en.prefix.csv" \
  --shard-glob "$SHARD_OUTPUT_DIR/shard_*.csv" \
  --out "$TRANSLATED_DIR/hindi_script_en.csv" \
  --lowcos-out "$TRANSLATED_DIR/hindi_script_lowcos.csv" \
  --expected-total "$expected_total" \
  --inspect-quantile "$INSPECT_QUANTILE" \
  --highrisk-quantile "$HIGHRISK_QUANTILE" \
  > "$LOG_DIR/06_assemble_native_multigpu.log" 2>&1

PYTHONUNBUFFERED=1 micromamba run -n "$ENV_NAME" \
  python "$BASE_DIR/merge_translation_outputs.py" \
  --english "$ROUTED_DIR/english_tweets.csv" \
  --native-translated "$TRANSLATED_DIR/hindi_script_en.csv" \
  --roman-translated "$TRANSLATED_DIR/hinglish_roman_en.csv" \
  --output "$RUN_DIR/retweet_data_combined_labeled_by_retweet_en.csv" \
  --text-col "$TEXT_COL" \
  > "$LOG_DIR/07_merge_after_multigpu.log" 2>&1

echo "final_output=$RUN_DIR/retweet_data_combined_labeled_by_retweet_en.csv"
