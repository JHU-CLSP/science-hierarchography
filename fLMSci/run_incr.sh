#!/bin/bash
###############################################################################
# fLMSci • Incremental Topic‑Placement Pipeline
#   Step 1  topic_rationale_gen.py   → topics_rationales.csv + unique_topics.txt
#   Step 2  incr/code/llm_incr.py    → incremental taxonomy
#   Step 3  map_papers.py            → add papers to taxonomy
#   Optional evaluation             → evaluation.py
###############################################################################

### 0. Robust Paths ###########################################################
SCRIPT_DIR="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"   # …/fLMSci

#  top‑level scripts
TL_DIR="$SCRIPT_DIR"

#  incr script location
INCR_DIR="$SCRIPT_DIR/incr"
if [ -d "$INCR_DIR/code" ]; then
  INCR_CODE_DIR="$INCR_DIR/code"
else
  INCR_CODE_DIR="$INCR_DIR"
fi

PROMPT_DIR="$SCRIPT_DIR/prompts"          # reuse the same topic prompt
RAW_JSON_DIR="$SCRIPT_DIR/jsons"              # raw papers come from here
RESULTS_DIR="$SCRIPT_DIR/results"        # all outputs go here
mkdir -p "$RESULTS_DIR" "$RESULTS_DIR/abstracts" "$RESULTS_DIR/incr/taxonomy"

### 1. Defaults ################################################################
EVALUATE=false
CSV_PATH="$RESULTS_DIR/topics_rationales.csv"   # adjust if needed
OUTPUT_EVAL_PATH="$RESULTS_DIR/incr/evaluation_output_incr.xlsx"
SAMPLE_SIZE=200

# llm_incr default knobs
BATCH_SIZE=32
MAX_DEPTH=10
MAX_TOKENS=32000
MAX_RESP_TOKENS=256

### 2. CLI Arguments ###########################################################
while [[ "$#" -gt 0 ]]; do
  case $1 in
      --evaluate)       EVALUATE=true ;;
      --csv)            CSV_PATH="$2";            shift ;;
      --output)         OUTPUT_EVAL_PATH="$2";    shift ;;
      --sample_size)    SAMPLE_SIZE="$2";         shift ;;
      --batch_size)     BATCH_SIZE="$2";          shift ;;
      --max_depth)      MAX_DEPTH="$2";           shift ;;
      --max_tokens)     MAX_TOKENS="$2";          shift ;;
      --max_resp)       MAX_RESP_TOKENS="$2";     shift ;;
      *) echo "Unknown flag: $1"; exit 1 ;;
  esac
  shift
done

### 3. Sanity Checks ###########################################################
for f in topic_rationale_gen.py map_papers.py; do
  test -f "$TL_DIR/$f" || { echo "❌ $f not found in $TL_DIR"; exit 1; }
done
test -f "$INCR_CODE_DIR/llm_incr.py" || {
  echo "❌ llm_incr.py not found in $INCR_CODE_DIR"; exit 1; }
test -f "$PROMPT_DIR/topic_gen.txt" || {
  echo "❌ Prompt missing: $PROMPT_DIR/topic_gen.txt"; exit 1; }

###############################################################################
echo "[STEP 1] Generating Topics and Rationales"
python "$TL_DIR/topic_rationale_gen.py" \
  --input_folder  "$RAW_JSON_DIR" \
  --output_folder "$RESULTS_DIR/abstracts" \
  --prompt_path   "$PROMPT_DIR/topic_gen.txt" \
  --output_csv    "$RESULTS_DIR/topics_rationales.csv" \
  --model gpt-4 \
  --topics_txt    "$RESULTS_DIR/unique_topics.txt"

echo "[STEP 2] Incremental Topic Placement"
python "$INCR_CODE_DIR/llm_incr.py" \
  --topics_path   "$RESULTS_DIR/unique_topics.txt" \
  --taxonomy_path "$SCRIPT_DIR/taxonomy/science_seed.json" \
  --results_dir   "$RESULTS_DIR/incr" \
  --batch_size    "$BATCH_SIZE" \
  --max_depth     "$MAX_DEPTH" \
  --max_tokens    "$MAX_TOKENS" \
  --max_response_tokens "$MAX_RESP_TOKENS"

FINAL_TAX_JSON="$RESULTS_DIR/incr/taxonomy/final_taxonomy_llm_incr.json"
if [ ! -f "$FINAL_TAX_JSON" ]; then
  echo "❌ Expected taxonomy not found: $FINAL_TAX_JSON"
  echo "   Check llm_incr.py output path."
  exit 1
fi

echo "[STEP 3] Mapping Papers → Topics"
python "$TL_DIR/map_papers.py" \
  --csv_path      "$RESULTS_DIR/topics_rationales.csv" \
  --taxonomy_path "$FINAL_TAX_JSON" \
  --output_path   "$RESULTS_DIR/incr/topics_taxonomy_papers_incr.json"

### 4. Optional Evaluation #####################################################
if [ "$EVALUATE" = true ]; then
  echo "[OPTIONAL] Evaluation Step"
  python "$SCRIPT_DIR/evaluation.py" \
    --taxonomy   "$RESULTS_DIR/incr/topics_taxonomy_papers_incr.json" \
    --csv        "$CSV_PATH" \
    --output     "$OUTPUT_EVAL_PATH" \
    --sample_size "$SAMPLE_SIZE"
fi

echo "[DONE] Incremental pipeline finished successfully."
###############################################################################