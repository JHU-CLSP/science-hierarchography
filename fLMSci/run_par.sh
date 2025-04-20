#!/bin/bash
###############################################################################
# fLMSci • Parallel Topic‑Placement Pipeline
#  - top‑level:   topic_rationale_gen.py, map_papers.py, evaluation.py
#  - par/code/:   baseline.py, taxonomy_merge.py
###############################################################################

### 0.  Robust Base Paths ######################################################
SCRIPT_DIR="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"   # …/fLMSci
PAR_DIR="$SCRIPT_DIR/par"

# detect whether scripts are in par/code/ or par/
if [ -d "$PAR_DIR/code" ]; then
  CODE_DIR="$PAR_DIR/code"      # baseline.py , taxonomy_merge.py live here
else
  CODE_DIR="$PAR_DIR"
fi

TL_DIR="$SCRIPT_DIR"            # topic_rationale_gen.py , map_papers.py
PROMPT_DIR="$SCRIPT_DIR/prompts"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR" "$RESULTS_DIR/abstracts" "$RESULTS_DIR/par"

### 1.  Defaults ###############################################################
EVALUATE=false
CSV_PATH="$RESULTS_DIR/topics_rationales.csv"   # adjust if needed
OUTPUT_EVAL_PATH="$RESULTS_DIR/par/evaluation_output_par.xlsx"
SAMPLE_SIZE=200

### 2.  CLI Arguments ##########################################################
while [[ "$#" -gt 0 ]]; do
  case $1 in
      --evaluate)     EVALUATE=true ;;
      --csv)          CSV_PATH="$2";      shift ;;
      --output)       OUTPUT_EVAL_PATH="$2"; shift ;;
      --sample_size)  SAMPLE_SIZE="$2";   shift ;;
      *) echo "Unknown flag: $1"; exit 1 ;;
  esac
  shift
done

### 3.  Sanity Check ###########################################################
for f in topic_rationale_gen.py map_papers.py; do
  test -f "$TL_DIR/$f" || { echo "❌ $f not found in $TL_DIR"; exit 1; }
done
for f in baseline.py taxonomy_merge.py; do
  test -f "$CODE_DIR/$f" || { echo "❌ $f not found in $CODE_DIR"; exit 1; }
done
test -f "$PROMPT_DIR/topic_gen.txt" || {
  echo "❌ Prompt file not found: $PROMPT_DIR/topic_gen.txt"; exit 1; }

###############################################################################
echo "[STEP 1] Generating Topics and Rationales"
python "$TL_DIR/topic_rationale_gen.py" \
  --input_folder "$SCRIPT_DIR/jsons" \
  --output_folder "$RESULTS_DIR/abstracts" \
  --prompt_path "$PROMPT_DIR/topic_gen.txt" \
  --output_csv "$RESULTS_DIR/topics_rationales.csv" \
  --model gpt-4 \
  --topics_txt "$RESULTS_DIR/unique_topics.txt"

echo "[STEP 2] Parallel Topic Placement"
python "$CODE_DIR/baseline.py" \
  --topics_path "$RESULTS_DIR/unique_topics.txt" \
  --taxonomy_path "$SCRIPT_DIR/taxonomy/science_seed.json" \
  --output_path "$RESULTS_DIR/par/taxonomy_creation_outputs.xlsx" \
  --chunk_size 100 --max_iterations 2 --retries 1

echo "[STEP 3] Merging Taxonomies"
python "$CODE_DIR/taxonomy_merge.py" \
  --excel_path "$RESULTS_DIR/par/taxonomy_creation_outputs.xlsx" \
  --json_output "$RESULTS_DIR/par/final_taxonomy_par.json"

echo "[STEP 4] Mapping Papers → Topics"
python "$TL_DIR/map_papers.py" \
  --csv_path "$RESULTS_DIR/topics_rationales.csv" \
  --taxonomy_path "$RESULTS_DIR/par/final_taxonomy_par.json" \
  --output_path "$RESULTS_DIR/par/topics_taxonomy_papers_par.json"

### 5.  Optional Evaluation ####################################################
if [ "$EVALUATE" = true ]; then
  echo "[OPTIONAL] Evaluation Step"
  python "$SCRIPT_DIR/evaluation.py" \
    --taxonomy   "$RESULTS_DIR/topics_taxonomy_papers.json" \
    --csv        "$CSV_PATH" \
    --output     "$OUTPUT_EVAL_PATH" \
    --sample_size "$SAMPLE_SIZE"
fi

echo "[DONE] Pipeline finished successfully."
###############################################################################
