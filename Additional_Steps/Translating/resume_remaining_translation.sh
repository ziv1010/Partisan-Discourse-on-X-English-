#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${1:?usage: resume_remaining_translation.sh /abs/path/to/run_dir}"
ENV_NAME="${ENV_NAME:-x_translate}"
TEXT_COL="${TEXT_COL:-tweet}"

ROMAN_GPU="${ROMAN_GPU:-0}"
ROMAN_CHUNK_SIZE="${ROMAN_CHUNK_SIZE:-20000}"
ROMAN_BATCH_SIZE="${ROMAN_BATCH_SIZE:-16}"
INSPECT_QUANTILE="${INSPECT_QUANTILE:-0.10}"
HIGHRISK_QUANTILE="${HIGHRISK_QUANTILE:-0.05}"
POLL_SECONDS="${POLL_SECONDS:-120}"

ROUTED_DIR="$RUN_DIR/routed"
TRANSLATED_DIR="$RUN_DIR/translated"
LOG_DIR="$RUN_DIR/logs"

ROUTE_LOG="$LOG_DIR/01_route.log"
NATIVE_LOG="$LOG_DIR/02_translate_native.log"
ROMAN_LOG="$LOG_DIR/03_translate_roman.log"
MERGE_LOG="$LOG_DIR/04_merge.log"

NATIVE_INPUT="$ROUTED_DIR/hindi_script_tweets.csv"
ROMAN_INPUT="$ROUTED_DIR/hinglish_roman_tweets.csv"
NATIVE_OUTPUT="$TRANSLATED_DIR/hindi_script_en.csv"
ROMAN_OUTPUT="$TRANSLATED_DIR/hinglish_roman_en.csv"
FINAL_OUTPUT="$RUN_DIR/retweet_data_combined_labeled_by_retweet_en.csv"

for required in "$ROUTE_LOG" "$NATIVE_INPUT" "$ROMAN_INPUT"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 1
  fi
done

mkdir -p "$TRANSLATED_DIR" "$LOG_DIR"

branch_expected_count() {
  local branch="$1"
  python3 - "$ROUTE_LOG" "$branch" <<'PY'
from pathlib import Path
import re
import sys

log_path = Path(sys.argv[1])
branch = sys.argv[2]
text = log_path.read_text(errors="ignore")
pat = re.compile(r"\[chunk \d+\] rows=\d+ total=\d+ english=(\d+) native=(\d+) roman=(\d+)")
idx = {"english": 0, "native": 1, "roman": 2}[branch]
total = 0
for match in pat.findall(text):
    total += int(match[idx])
print(total)
PY
}

log_last_total() {
  local log_path="$1"
  python3 - "$log_path" <<'PY'
from pathlib import Path
import re
import sys

log_path = Path(sys.argv[1])
if not log_path.exists():
    print(0)
    raise SystemExit
text = log_path.read_text(errors="ignore")
matches = re.findall(r"chunk \d+ processed \d+ rows; total (\d+)", text)
print(matches[-1] if matches else 0)
PY
}

active_process_count() {
  local pattern="$1"
  local count
  count=$(pgrep -af "$pattern" | wc -l || true)
  echo "${count//[[:space:]]/}"
}

wait_for_pattern() {
  local pattern="$1"
  local label="$2"
  local log_path="$3"
  local expected="$4"
  while [[ "$(active_process_count "$pattern")" -gt 0 ]]; do
    local current
    current=$(log_last_total "$log_path")
    echo "[wait] $label running: current_total=$current expected_total=$expected"
    sleep "$POLL_SECONDS"
  done
}

expected_native=$(branch_expected_count native)
expected_roman=$(branch_expected_count roman)

native_pattern="full_run_hindi_pipeline.py --input $NATIVE_INPUT"
roman_pattern="full_run_hindi_pipeline.py --input $ROMAN_INPUT"

echo "run_dir=$RUN_DIR"
echo "expected_native=$expected_native"
echo "expected_roman=$expected_roman"

roman_total_before=$(log_last_total "$ROMAN_LOG")
if [[ "$expected_roman" -gt 0 && "$roman_total_before" -lt "$expected_roman" ]]; then
  if [[ "$(active_process_count "$roman_pattern")" -gt 0 ]]; then
    echo "[info] roman translation already running"
  else
    if [[ -f "$ROMAN_LOG" ]]; then
      mv "$ROMAN_LOG" "$LOG_DIR/03_translate_roman.failed_attempt.log"
    fi
    if [[ -f "$ROMAN_OUTPUT" ]]; then
      mv "$ROMAN_OUTPUT" "$TRANSLATED_DIR/hinglish_roman_en.failed_attempt.csv"
    fi
    if [[ -f "$TRANSLATED_DIR/hinglish_roman_lowcos.csv" ]]; then
      mv "$TRANSLATED_DIR/hinglish_roman_lowcos.csv" \
        "$TRANSLATED_DIR/hinglish_roman_lowcos.failed_attempt.csv"
    fi

    echo "[start] rerunning roman translation on gpu=$ROMAN_GPU batch_size=$ROMAN_BATCH_SIZE"
    PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES="$ROMAN_GPU" \
      micromamba run -n "$ENV_NAME" \
      python "$BASE_DIR/full_run_hindi_pipeline.py" \
      --input "$ROMAN_INPUT" \
      --text-col "$TEXT_COL" \
      --src-lang-col nllb_src_lang \
      --engine nllb \
      --out-translated "$ROMAN_OUTPUT" \
      --lowcos-out "$TRANSLATED_DIR/hinglish_roman_lowcos.csv" \
      --chunk-size "$ROMAN_CHUNK_SIZE" \
      --batch-size "$ROMAN_BATCH_SIZE" \
      --inspect-quantile "$INSPECT_QUANTILE" \
      --highrisk-quantile "$HIGHRISK_QUANTILE" \
      > "$ROMAN_LOG" 2>&1 &
  fi
fi

if [[ "$(active_process_count "$native_pattern")" -gt 0 ]]; then
  wait_for_pattern "$native_pattern" "native translation" "$NATIVE_LOG" "$expected_native"
fi

if [[ "$(active_process_count "$roman_pattern")" -gt 0 ]]; then
  wait_for_pattern "$roman_pattern" "roman translation" "$ROMAN_LOG" "$expected_roman"
fi

native_total_after=$(log_last_total "$NATIVE_LOG")
roman_total_after=$(log_last_total "$ROMAN_LOG")

echo "[check] native_total_after=$native_total_after"
echo "[check] roman_total_after=$roman_total_after"

if [[ "$native_total_after" -ne "$expected_native" ]]; then
  echo "Native translation incomplete: expected $expected_native, got $native_total_after" >&2
  exit 1
fi

if [[ "$expected_roman" -gt 0 && "$roman_total_after" -ne "$expected_roman" ]]; then
  echo "Roman translation incomplete: expected $expected_roman, got $roman_total_after" >&2
  exit 1
fi

rm -f "$FINAL_OUTPUT"
PYTHONUNBUFFERED=1 micromamba run -n "$ENV_NAME" \
  python "$BASE_DIR/merge_translation_outputs.py" \
  --english "$ROUTED_DIR/english_tweets.csv" \
  --native-translated "$NATIVE_OUTPUT" \
  --roman-translated "$ROMAN_OUTPUT" \
  --output "$FINAL_OUTPUT" \
  --text-col "$TEXT_COL" \
  > "$MERGE_LOG" 2>&1

echo "final_output=$FINAL_OUTPUT"
