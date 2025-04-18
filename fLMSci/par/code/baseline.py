import json
import os
import random
import re
import time
import argparse
import gc
import pandas as pd
from vllm import LLM, SamplingParams
import multiprocessing as mp
if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)    # must be first!
import torch  # safe only after spawn is set

import torch
from transformers import AutoTokenizer

#torch.cuda.empty_cache()

# -------------------------------
# Argument Parser Setup
# -------------------------------
parser = argparse.ArgumentParser(description="Taxonomy placement using LLaMA.")
parser.add_argument('--topics_path', type=str, required=True, help="Path to topics file")
parser.add_argument('--taxonomy_path', type=str, required=True, help="Path to seed taxonomy JSON file")
parser.add_argument('--output_path', type=str, required=True, help="Output Excel file to save results")
parser.add_argument('--chunk_size', type=int, default=100, help="Number of topics per chunk")
parser.add_argument('--max_iterations', type=int, default=2, help="Max retry iterations per chunk")
parser.add_argument('--retries', type=int, default=1, help="Max LLM retries per attempt")

args = parser.parse_args()

# -------------------------------
# Load Model & Tokenizer
# -------------------------------
print("Loading tokenizer and model. This may take a while...", flush=True)
model_id = "meta-llama/Llama-3.3-70B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
llm = LLM(
    model=model_id,
    tensor_parallel_size=4,  # Adjust based on the number of GPUs available
    dtype="bfloat16",
    gpu_memory_utilization=0.85,  # Adjust to a lower value if memory issues persist
    max_model_len=5000  # Set based on the maximum context length you intend to use
)
# one shared sampling profile for every call
SAMPLING = SamplingParams(
    temperature = 0.0,     # deterministic (set >0 for creative)
    top_p       = 1.0,
    max_tokens  = 50000,    # matches previous cap (prompt‑side len checked below)
    stop=["<|eot_id|>"]
)
print("✅  Model ready.", flush=True)
print("Model and pipeline loaded successfully.", flush=True)

# -------------------------------
# Utility: Text Generation
# -------------------------------
def llama_generate(prompt: str, sampling_params: SamplingParams = SAMPLING) -> str:
    """
    Thin wrapper around vLLM.generate().
    Returns the *raw* assistant text (no JSON trimming).
    """
    # safety: truncate overly long prompts to leave room for reply
    prompt_ids = tokenizer.encode(prompt)
    # if len(prompt_ids) > args.max_ctx - sampling_params.max_tokens:
    #     raise ValueError(
    #         f"Prompt length {len(prompt_ids)} exceeds model context window "
    #         f"({args.max_ctx}). Consider increasing --max_ctx or shrinking "
    #         f"--chunk_size."
    #     )

    out = llm.generate(prompt, sampling_params)[0]
    return out.outputs[0].text.strip()

# -------------------------------
# Load and Prepare Topics
# -------------------------------
with open(args.topics_path, 'r', encoding='utf-8') as f:
    unique_topics = list(set(line.strip() for line in f.readlines()))

random.shuffle(unique_topics)

def chunk_topics(topics, chunk_size):
    for i in range(0, len(topics), chunk_size):
        yield topics[i:i + chunk_size]

chunked_topics = list(chunk_topics(unique_topics, chunk_size=args.chunk_size))
print(f'Total chunks created: {len(chunked_topics)}', flush=True)

# -------------------------------
# Taxonomy Utils
# -------------------------------
def extract_all_nodes(taxonomy_dict):
    nodes = set()
    def traverse(d):
        for key, value in d.items():
            nodes.add(key)
            if isinstance(value, dict): traverse(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict): traverse(item)
    traverse(taxonomy_dict)
    return nodes

def validate_topics_in_taxonomy(topics, taxonomy_json_str):
    try:
        taxonomy_dict = json.loads(taxonomy_json_str)
        taxonomy_nodes = extract_all_nodes(taxonomy_dict)
        matched = [t for t in topics if t in taxonomy_nodes]
        missing = [t for t in topics if t not in taxonomy_nodes]
        return matched, missing
    except json.JSONDecodeError:
        print("Invalid JSON in taxonomy.", flush=True)
        return [], topics

# -------------------------------
# Load Seed Taxonomy
# -------------------------------
with open(args.taxonomy_path, 'r') as f:
    seed_taxonomy = json.load(f)

# -------------------------------
# Core Model Processing
# -------------------------------
def process_chunk(topics_chunk, seed_taxonomy, chunk_id, retries, prompt):
    calls_made = 0
    tokens_used = 0
    for attempt in range(retries):
        try:
            calls_made += 1
            chat_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": f'''{prompt}'''}],
                add_generation_prompt=True,
                tokenize=False
            )
            resp = llama_generate(chat_prompt, sampling_params=SAMPLING)
            prompt_tokens = len(tokenizer.encode(prompt))
            response_tokens = len(tokenizer.encode(resp))
            tokens_used += (prompt_tokens + response_tokens)
            print(f"Printing response: {resp}", flush=True)
            json_match = re.search(r'```json(.*?)```', resp, re.DOTALL)
            reply_content = json_match.group(1).strip() if json_match else resp

            if reply_content:
                matched, missing = validate_topics_in_taxonomy(topics_chunk, reply_content)
                return reply_content, missing, calls_made, tokens_used

        except Exception as e:
            print(f"Error processing chunk {chunk_id} attempt {attempt+1}: {e}", flush=True)
            time.sleep(2)

    return None, topics_chunk, calls_made, tokens_used

# -------------------------------
# Wrapper: Retry Missing Topics
# -------------------------------
def process_chunk_wrapper(chunk_id, topics_chunk, prompt):
    current_taxonomy = seed_taxonomy
    topics_for_iteration = topics_chunk[:]
    final_taxonomy_str = None
    final_missing = None
    total_calls = 0
    total_tokens = 0

    for iteration in range(1, args.max_iterations + 1):
        print(f"Processing chunk {chunk_id} - Iteration {iteration}", flush=True)
        taxonomy_str, missing, calls_made, tokens_used = process_chunk(topics_for_iteration, current_taxonomy, chunk_id, args.retries, prompt)
        total_calls += calls_made
        total_tokens += tokens_used

        if taxonomy_str is None:
            print(f"Chunk {chunk_id} failed on iteration {iteration}", flush=True)
            break

        try:
            current_taxonomy = json.loads(taxonomy_str)
        except json.JSONDecodeError:
            print(f"Chunk {chunk_id}: Invalid JSON. Retrying...", flush=True)
            continue

        _, still_missing = validate_topics_in_taxonomy(topics_for_iteration, taxonomy_str)

        if not still_missing:
            print(f"Chunk {chunk_id}: All topics placed.", flush=True)
            final_missing = []
            final_taxonomy_str = taxonomy_str
            break
        else:
            print(f"Chunk {chunk_id}: {len(still_missing)} topics still missing...", flush=True)
            topics_for_iteration = still_missing
            final_taxonomy_str = taxonomy_str
            final_missing = still_missing

    return {
        "chunk_id": chunk_id,
        "original_chunked_topics": topics_chunk,
        "implemented_taxonomy": final_taxonomy_str,
        "missing_topics": final_missing if final_missing else [],
        "llm_calls_made": total_calls,
        "total_tokens_used": total_tokens
    }

# -------------------------------
# Process All Chunks
# -------------------------------
# read prompt from file
with open('science-cartography/fLMSci/prompts/parallel_placement.txt', 'r') as file:
    prompt = file.read()
results_list = []
for chunk_id, chunk in enumerate(chunked_topics, start=1):
    result = process_chunk_wrapper(chunk_id, chunk, prompt)
    results_list.append(result)
    print(f"Finished chunk {chunk_id}", flush=True)

    if chunk_id % 5 == 0:
        pd.DataFrame(results_list).to_excel(args.output_path, index=False)
        print(f"Checkpoint: Saved to {args.output_path}", flush=True)

# Final save
pd.DataFrame(results_list).to_excel(args.output_path, index=False)
print(f"All chunks processed. Final results saved to {args.output_path}.", flush=True)
del llm
gc.collect()
torch.cuda.empty_cache()  
# Example usage: 
# python science-cartography/fLMSci/par/baseline.py \
#     --topics_path /path/to/unique_topics.txt \
#     --taxonomy_path /path/to/science_seed.json \
#     --output_path /path/to/results.xlsx \
#     --chunk_size 100 \
#     --max_iterations 2 \
#     --retries 1