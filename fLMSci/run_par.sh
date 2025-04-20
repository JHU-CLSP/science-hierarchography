#!/bin/bash
###############################################################################
# fLMSci • Parallel Topic‑Placement Pipeline
###############################################################################

### 0. Robust Paths ###########################################################
SCRIPT_DIR="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"   # …/fLMSci

PAR_DIR="$SCRIPT_DIR/par"
PAR_CODE_DIR="$PAR_DIR/code"
PROMPT_DIR="$SCRIPT_DIR/prompts"
RAW_JSON_DIR="$SCRIPT_DIR/jsons"
RESULTS_DIR="$SCRIPT_DIR/results"

mkdir -p "$RESULTS_DIR" "$RESULTS_DIR/abstracts" "$RESULTS_DIR/par"

### 1. Defaults ################################################################
EVALUATE=false
CSV_PATH="$RESULTS_DIR/topics_rationales.csv"
OUTPUT_EVAL_PATH="$RESULTS_DIR/par/evaluation_output_par.xlsx"
SAMPLE_SIZE=2
CHUNK_SIZE=100
MAX_ITER=2
RETRIES=1

### 2. CLI Arguments ###########################################################
while [[ "$#" -gt 0 ]]; do
  case $1 in
      --evaluate)     EVALUATE=true ;;
      --csv)          CSV_PATH="$2";         shift ;;
      --output)       OUTPUT_EVAL_PATH="$2"; shift ;;
      --sample_size)  SAMPLE_SIZE="$2";      shift ;;
      --chunk_size)   CHUNK_SIZE="$2";       shift ;;
      --max_iter)     MAX_ITER="$2";         shift ;;
      --retries)      RETRIES="$2";          shift ;;
      *) echo "Unknown flag: $1"; exit 1 ;;
  esac
  shift
done

### 3. Sanity Checks ###########################################################
for f in topic_rationale_gen.py map_papers.py; do
  test -f "$SCRIPT_DIR/$f" || { echo "❌ $f not found in $SCRIPT_DIR"; exit 1; }
done
for f in baseline.py taxonomy_merge.py; do
  test -f "$PAR_CODE_DIR/$f" || { echo "❌ $f not found in $PAR_CODE_DIR"; exit 1; }
done
test -f "$PROMPT_DIR/topic_gen.txt" || {
  echo "❌ Missing prompt file: $PROMPT_DIR/topic_gen.txt"; exit 1; }

###############################################################################
echo "[STEP 1] Generating Topics and Rationales"
python "$SCRIPT_DIR/topic_rationale_gen.py" \
  --input_folder "$RAW_JSON_DIR" \
  --output_folder "$RESULTS_DIR/abstracts" \
  --prompt_path "$PROMPT_DIR/topic_gen.txt" \
  --output_csv "$CSV_PATH" \
  --model gpt-4 \
  --topics_txt "$RESULTS_DIR/unique_topics.txt"

echo "[STEP 2] Parallel Topic Placement"
python "$PAR_CODE_DIR/baseline.py" \
  --topics_path "$RESULTS_DIR/unique_topics.txt" \
  --taxonomy_path "$SCRIPT_DIR/taxonomy/science_seed.json" \
  --prompt_path "$PAR_DIR/prompts/parallel_placement.txt" \
  --output_path "$RESULTS_DIR/par/taxonomy_creation_outputs.xlsx" \
  --chunk_size "$CHUNK_SIZE" \
  --max_iterations "$MAX_ITER" \
  --retries "$RETRIES"

echo "[STEP 3] Merging Taxonomies"
python "$PAR_CODE_DIR/taxonomy_merge.py" \
  --excel_path "$RESULTS_DIR/par/taxonomy_creation_outputs.xlsx" \
  --json_output "$RESULTS_DIR/par/final_taxonomy_par.json"

echo "[STEP 4] Mapping Papers → Topics"
python "$SCRIPT_DIR/map_papers.py" \
  --csv_path "$CSV_PATH" \
  --taxonomy_path "$RESULTS_DIR/par/final_taxonomy_par.json" \
  --output_path "$RESULTS_DIR/par/topics_taxonomy_papers_par.json"

### 5. Optional Evaluation #####################################################
if [ "$EVALUATE" = true ]; then
  echo "[OPTIONAL] Evaluation Step"
  python "$SCRIPT_DIR/evaluation.py" \
    --taxonomy "$RESULTS_DIR/par/topics_taxonomy_papers_par.json" \
    --csv "$CSV_PATH" \
    --output "$OUTPUT_EVAL_PATH" \
    --sample_size "$SAMPLE_SIZE"
fi

echo "[DONE] Parallel pipeline finished successfully."
###############################################################################