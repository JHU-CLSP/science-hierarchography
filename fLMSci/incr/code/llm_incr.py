import json
import pandas as pd
import torch
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from pathlib import Path
import argparse

torch.cuda.empty_cache()

# =============================
#        CONFIGURATION
# =============================

parser = argparse.ArgumentParser(description="Taxonomy placement(incremental) using Llama.")
parser.add_argument('--topics_path', type=str, required=True, help="Path to unique_topics.txt")
parser.add_argument('--taxonomy_path', type=str, required=True, help="Path to seed taxonomy JSON file")
parser.add_argument('--results_dir', type=str, required=True, help="Path for results directory")
parser.add_argument('--batch_size', type=int, default=32, help="Batch size for LLM calls")
parser.add_argument('--max_depth', type=int, default=10, help="Max depth traversals")
parser.add_argument('--max_tokens', type=int, default=32000, help="Max tokens for LLM")
parser.add_argument('--max_response_tokens', type=int, default=256, help="Max response tokens for LLM")
parser.add_argument('--load_checkpoint', type=bool, default=False, help="Load checkpointed taxonomy if path exists")
args = parser.parse_args()

RESULTS_FILE = f"{args.results_dir}/excel/final_results_llm_incr.xlsx"

with open(args.topics_path, "r") as f:
    TOPICS = [line.strip() for line in f if line.strip()]

TOPICS = TOPICS[:200]
    
SUBNODE_DESCRIPTIONS = {
    "Formal Sciences": "Focuses on abstract systems and formal methodologies grounded in logic, mathematics, and symbolic reasoning. Provides theoretical frameworks (e.g., statistics, computer science, systems theory) used to model and solve problems across empirical disciplines and technology.",
    "Natural Sciences": "Investigates the physical universe and living organisms through empirical observation, experimentation, and theoretical analysis. Includes physical sciences (e.g., physics, chemistry, astronomy) and biological sciences (e.g., genetics, ecology) to uncover fundamental laws and processes governing nature.",
    "Social Sciences": "Studies human behavior, societies, and institutions using qualitative and quantitative methods. Encompasses disciplines like psychology, economics, and political science to analyze cultural, economic, and social interactions within historical and geographic contexts."
}

# =============================
#       TOKEN BUDGET LOGIC
# =============================

# MAX_TOTAL_TOKENS = 32000
# MAX_RESPONSE_TOKENS = 256
MAX_PROMPT_TOKENS = args.max_tokens - args.max_response_tokens

def estimate_base_prompt_tokens(tokenizer, current_path, new_topic, is_leaf):
    dummy_subnodes = '["TEMP_PLACEHOLDER_SUBNODE"]'
    prompt = generate_prompt(current_path, dummy_subnodes, new_topic, is_leaf)
    encoded = tokenizer.encode(prompt)
    return len(encoded)

def truncate_subnodes_to_fit(tokenizer, current_path, new_topic, is_leaf, all_subnodes):
    base_tokens = estimate_base_prompt_tokens(tokenizer, current_path, new_topic, is_leaf)
    remaining_tokens = MAX_PROMPT_TOKENS - base_tokens

    subnodes_included = []
    current_subnode_token_sum = 0

    for sn in all_subnodes:
        if sn in SUBNODE_DESCRIPTIONS:
            formatted = f'"{sn}": "{SUBNODE_DESCRIPTIONS[sn]}"'
        else:
            formatted = f'"{sn}"'

        token_len = len(tokenizer.encode(formatted))
        if current_subnode_token_sum + token_len > remaining_tokens:
            break
        subnodes_included.append(sn)
        current_subnode_token_sum += token_len

    return subnodes_included


# Global tracking
call_count = {}
token_count = {}
results = []
DISCARDED_TOPICS = {}


# =============================
#        LOAD MODEL
# =============================

def load_vllm_model():
    model_id = "meta-llama/Llama-3.3-70B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm = LLM(
        model=model_id,
        tensor_parallel_size=4,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        max_model_len=33000,
        device="cuda",
        enforce_eager=True,
    )
    return llm, tokenizer

# =============================
#     TAXONOMY UTILITIES
# =============================

def load_taxonomy_from_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def get_subnodes_at_path(taxonomy_dict, path):
    current = taxonomy_dict
    for p in path:
        current = current.get(p, {})
    return list(current.keys()) if isinstance(current, dict) else []

def go_down(path, node):
    return path + [node]

def path_exists(taxonomy_dict, path):
    current = taxonomy_dict
    for step in path:
        if step not in current:
            return False
        current = current[step]
    return True

def add_sibling(taxonomy_dict, current_path, new_node):
    current = taxonomy_dict
    for step in current_path:
        if step not in current:
            print(f"❌ [ERROR] Path step '{step}' not found. Cannot add sibling '{new_node}' at {current_path}", flush=True)
            return
        current = current[step]

    if new_node not in current:
        current[new_node] = {}
        print(f"✅ Added sibling '{new_node}' at path {current_path}", flush=True)


def add_as_child(taxonomy_dict, current_path, parent_node, new_node):
    current = taxonomy_dict
    for step in current_path:
        if step not in current:
            return False
        current = current[step]
    if parent_node not in current or not isinstance(current[parent_node], dict):
        return False
    if new_node not in current[parent_node]:
        current[parent_node][new_node] = {}
    return True


def make_parent_for_multiple(taxonomy_dict, current_path, parent_node, children):
    current = taxonomy_dict
    for step in current_path:
        if step not in current:
            print(f"❌ Invalid path while making parent: {current_path}",flush=True)
            return False
        current = current[step]

    # Avoid overwriting existing nodes silently
    if parent_node not in current:
        current[parent_node] = {}

    parent_dict = current[parent_node]
    moved = []

    for child in children:
        if child not in current:
            print(f"⚠️ Child '{child}' not found at path {current_path}. Skipping.",flush=True)
            continue
        parent_dict[child] = current.pop(child)
        moved.append(child)

    if not moved:
        print(f"⚠️ No children moved under '{parent_node}'. Nothing changed.",flush=True)
        return False

    print(f"✅ Created parent '{parent_node}' and moved children {moved} under it.",flush=True)
    return True


def discard_topic(topic, reason):
    print(f"[DISCARDED] {topic}: {reason}",flush=True)

def extract_and_parse_json_blocks(text):
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)  # Fix escaping
    if match:
        try:
            return [json.loads(match.group(1))]
        except json.JSONDecodeError:
            pass
    try:
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start != -1 and json_end != -1:
            return [json.loads(text[json_start:json_end])]
    except Exception:
        pass
    return None

# =============================
#        PROMPT GENERATION
# =============================

def generate_prompt(current_path, subnodes, new_topic, is_leaf,force_go_down=False):
    response_format = '''{
    "action":
    '''
    if force_go_down:
        # Restrict LLM to only use `go_down` before depth 3
        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
You are building a scientific topics based taxonomy.

Path traced until now: {current_path}
Available subtopics at this level:
    subnodes = [{subnodes}]
New topic: "{new_topic}"

Your task is to identify which one of the existing subnodes is the best container for this topic.

The only valid action is:

1) "go_down"
   - "node": must be the name of one of the listed subnodes:[{subnodes}]
   - "explanation": optional

Your output must be valid JSON only:
{{
  "action": "go_down",
  "node": "string",
  "explanation": "string (optional)"
}}
No extra text.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```json
{response_format}
"""
        return prompt
    if is_leaf:
        # Leaf node prompt
        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
You are building a scientific topics based taxonomy.
Path traced until now: {current_path}
Available subtopics at this level:   
    subnodes = [{subnodes}]
New topic: "{new_topic}"

Evaluate all possible actions listed below equally before choosing the most appropriate one.
Avoid using `"add_sibling"` at very abstract levels like the root unless absolutely necessary. Prefer `"go_down"` or `"make_parent"` when appropriate and make the hierarchy meaningful.
Choose the action that best preserves a logical hierarchy, semantic clarity, and appropriate abstraction level.

Possible actions (unordered, no preference):
1) "add_sibling" – Use this if the appropriate path for the topic: {new_topic} is {current_path} and you want to place it here.
2) "discard" – Use this if the topic: {new_topic} is irrelevant, redundant, or already captured under another topic.

### Example of desired usage for each action:
1) "add_sibling"
   - "node": {new_topic} (the string we are adding at this level)
   - "parent_node": {current_path[-1]}
   - "explanation": optional
   - "child_nodes": not used.

2) "discard"
   - "node": {new_topic}
   - "explanation": optional
   - "parent_node", "child_nodes": not used

Your output must be valid JSON only:
{{
  "action": "add_sibling"|"discard",
  "node": "string",  
  "child_nodes": ["string", ...],  // only used if action=make_parent
  "explanation": "string (optional)"
}}
No extra text.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```json
{response_format}
"""
    else:
        # Non-leaf node
        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
You are building a scientific topics based taxonomy.

Path traced until now: {current_path}  
Subnode options available at this level: 
    subnodes = [{subnodes}]
New topic: "{new_topic}"  

Evaluate all possible actions listed below equally before choosing the most appropriate one.  
Choose the action that best preserves a logical hierarchy, semantic clarity, and appropriate abstraction level.

**Priority Guidance**:
1. FIRST consider "go_down" if ANY existing subnode could reasonably contain the new topic as a specialization
2. THEN consider "make_parent" if multiple existing subnodes could be grouped under a new category
3. ONLY use "add_sibling" if the topic is FUNDAMENTALLY distinct from all existing subnodes at this level
4. "discard" should be used for low-quality or redundant topics

**Critical Rules**:
- A node about "Applications of X" should ALWAYS go under X, not as a sibling
- Specific methods/tools belong under their parent field (e.g., "PCR" under "Molecular Biology")
- Avoid creating flat structures

Possible actions:

1) "go_down" – Use this if the topic: {new_topic} is a *more specific* subtype of one of the "subnodes" and belongs *within* it.
2) "add_sibling" – Use this if the topic: {new_topic} is on the same level of abstraction as the existing "subnodes". It should be added *alongside* them as a direct child of `{current_path[-1]}`.
3) "discard" – Use this if the topic: {new_topic} is irrelevant, redundant, or already captured under another topic.
4) "make_parent" – Use this when the new topic: {new_topic} or any one of the "subnodes" is broader or more abstract than one or more of the subnodes. In that case, make the new topic a direct child of `{current_path[-1]}` and move the relevant subnodes under it. Return them in `"child_nodes": [...]`.

### Example of desired usage for each action:
1) "go_down"
   - "node": must be the name of one of the existing "subnodes"
   - "explanation": an optional text with reasoning
   - "child_nodes", "parent_node": not used.

2) "add_sibling"
   - "node": {new_topic}
   - "parent_node": {current_path[-1]}
   - "explanation": optional
   - "child_nodes": not used.

3) "discard"
   - "node": {new_topic}
   - "explanation": optional
   - "parent_node", "child_nodes": not used

4) "make_parent"
   - "node": {new_topic} or one of the "subnodes" whichever is more appropriate
   - "child_nodes": array of the subnodes that must be moved under the new node
   - "explanation": optional
   - "parent_node": not used

Your output must be valid JSON only:
{{
  "action": "go_down"|"add_sibling"|"make_parent"|"discard",
  "node": "string",
  "parent_node": "string or null",  // only used if action = add_as_child
  "child_nodes": ["string", ...],   // only used if action = make_parent
  "explanation": "string (optional)"
}}
No extra text.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```json
{response_format}
"""
    return prompt

# =============================
#     DECISION APPLICATION
# =============================

def apply_decision(topic, entry, subnodes, is_leaf, decision, taxonomy):
    path = entry["path"]
    action = decision.get("action")
    node = decision.get("node")
    parent_node = decision.get("parent_node")
    child_nodes = decision.get("child_nodes", [])
    explanation = decision.get("explanation", "")

    placed = False

    # ========== LEAF LOGIC ==========
    if is_leaf:
        if action == "add_sibling" and path_exists(taxonomy, path):
            add_sibling(taxonomy, path, node)
            placed = True
        elif action == "discard":
            discard_topic(topic, explanation)
            results.append({
                "Topic": topic,
                "Action": action,
                "Node": node,
                "Parent_Node": parent_node,
                "Child_Nodes": child_nodes,
                "Explanation": explanation,
                "Final_Path": str(path),
                "Placed": False,
                "LLM_Calls": call_count.get(topic, 0),
                "Total_Tokens": token_count.get(topic, 0)
            })
            return True
        else:
            discard_topic(topic, f"Invalid leaf action '{action}' or path missing")
            results.append({
                "Topic": topic,
                "Action": "error",
                "Node": node,
                "Parent_Node": parent_node,
                "Child_Nodes": child_nodes,
                "Explanation": f"Invalid leaf action '{action}'",
                "Final_Path": str(path),
                "Placed": False,
                "LLM_Calls": call_count.get(topic, 0),
                "Total_Tokens": token_count.get(topic, 0)
            })
            return True

    # ========== NON-LEAF LOGIC ==========
    if entry["depth"] < 3 and not is_leaf:
        if action != "go_down":
            discard_topic(topic, f"Invalid action '{action}' at depth {entry['depth']} (must be 'go_down')")
            results.append({
                "Topic": topic,
                "Action": "discard",
                "Node": node,
                "Parent_Node": parent_node,
                "Child_Nodes": child_nodes,
                "Explanation": f"Invalid action at depth < 3",
                "Final_Path": str(path),
                "Placed": False,
                "LLM_Calls": call_count.get(topic, 0),
                "Total_Tokens": token_count.get(topic, 0)
            })
            return True

    if action == "go_down":
        new_path = go_down(path, node)
        if path_exists(taxonomy, new_path):
            entry["path"] = new_path
            entry["depth"] += 1
            # print(f"➡️ Topic '{topic}' going down to: {new_path}", flush=True)
            return False  # keep going deeper
        else:
            discard_topic(topic, f"go_down to non-existent subnode '{node}'")
            results.append({
                "Topic": topic,
                "Action": "discard",
                "Node": node,
                "Parent_Node": parent_node,
                "Child_Nodes": child_nodes,
                "Explanation": f"go_down to non-existent subnode '{node}'",
                "Final_Path": str(path),
                "Placed": False,
                "LLM_Calls": call_count.get(topic, 0),
                "Total_Tokens": token_count.get(topic, 0)
            })
            return True

    elif action == "add_sibling" and path_exists(taxonomy, path):
        add_sibling(taxonomy, path, node)
        placed = True

    elif action == "make_parent":
        if all(c in subnodes for c in child_nodes):
            success = make_parent_for_multiple(taxonomy, path, node, child_nodes)
            if success:
                if topic != node and add_as_child(taxonomy, path, node, topic):
                    placed = True
                elif topic == node:
                    placed = True
                else:
                    discard_topic(topic, f"Failed to attach topic '{topic}' under new parent '{node}'")
                    results.append({
                        "Topic": topic,
                        "Action": "discard",
                        "Node": node,
                        "Parent_Node": parent_node,
                        "Child_Nodes": child_nodes,
                        "Explanation": f"Failed to attach topic under new parent",
                        "Final_Path": str(path),
                        "Placed": False,
                        "LLM_Calls": call_count.get(topic, 0),
                        "Total_Tokens": token_count.get(topic, 0)
                    })
                    return True
            else:
                discard_topic(topic, f"make_parent failed — no children moved")
                results.append({
                    "Topic": topic,
                    "Action": "discard",
                    "Node": node,
                    "Parent_Node": parent_node,
                    "Child_Nodes": child_nodes,
                    "Explanation": f"make_parent failed — no children moved",
                    "Final_Path": str(path),
                    "Placed": False,
                    "LLM_Calls": call_count.get(topic, 0),
                    "Total_Tokens": token_count.get(topic, 0)
                })
                return True
        else:
            discard_topic(topic, f"make_parent failed — invalid children: {child_nodes}")
            results.append({
                "Topic": topic,
                "Action": "discard",
                "Node": node,
                "Parent_Node": parent_node,
                "Child_Nodes": child_nodes,
                "Explanation": f"make_parent failed — invalid children",
                "Final_Path": str(path),
                "Placed": False,
                "LLM_Calls": call_count.get(topic, 0),
                "Total_Tokens": token_count.get(topic, 0)
            })
            return True

    elif action == "discard":
        discard_topic(topic, explanation)
        results.append({
            "Topic": topic,
            "Action": action,
            "Node": node,
            "Parent_Node": parent_node,
            "Child_Nodes": child_nodes,
            "Explanation": explanation,
            "Final_Path": str(path),
            "Placed": False,
            "LLM_Calls": call_count.get(topic, 0),
            "Total_Tokens": token_count.get(topic, 0)
        })
        return True

    else:
        discard_topic(topic, f"Unrecognized action '{action}'")
        results.append({
            "Topic": topic,
            "Action": "discard",
            "Node": node,
            "Parent_Node": parent_node,
            "Child_Nodes": child_nodes,
            "Explanation": f"Unrecognized action",
            "Final_Path": str(path),
            "Placed": False,
            "LLM_Calls": call_count.get(topic, 0),
            "Total_Tokens": token_count.get(topic, 0)
        })
        return True

    # ✅ Log only final placement actions
    results.append({
        "Topic": topic,
        "Action": action,
        "Node": node,
        "Parent_Node": parent_node,
        "Child_Nodes": child_nodes,
        "Explanation": explanation,
        "Final_Path": str(path),
        "Placed": placed,
        "LLM_Calls": call_count.get(topic, 0),
        "Total_Tokens": token_count.get(topic, 0)
    })

    return placed or action in ("discard", "make_parent_invalid", "go_down_invalid")





# =============================
#     MAIN PLACEMENT LOOP
# =============================

import os

# Ensure these exist early in main()
os.makedirs(f"{args.results_dir}/taxonomy", exist_ok=True)
os.makedirs(f"{args.results_dir}/excel", exist_ok=True)

def multi_turn_batched_placement(active_topics, taxonomy, llm, tokenizer, processed_start):
    global call_count, token_count
    completed = []
    processed = processed_start
    placed_total = 0

    while active_topics:
        batch = active_topics[:args.batch_size]
        prompts = []
        metas = []

        for entry in batch:
            topic = entry["topic"]
            path = entry["path"]
            subnodes = get_subnodes_at_path(taxonomy, path)
            is_leaf = len(subnodes) == 0

            subnodes_sample = truncate_subnodes_to_fit(tokenizer, path, topic, is_leaf, subnodes)

            formatted = ", ".join(
                f'"{s}": "{SUBNODE_DESCRIPTIONS.get(s, "")}"' if s in SUBNODE_DESCRIPTIONS else f'"{s}"'
                for s in subnodes_sample
            )

            force_go_down = entry["depth"] < 3 and not is_leaf
            prompt = generate_prompt(path, formatted, topic, is_leaf, force_go_down=force_go_down)

            chat_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False
            )
            prompts.append(chat_prompt)
            metas.append({"entry": entry, "subnodes": subnodes, "is_leaf": is_leaf})

        try:
            outputs = llm.generate(
                prompts,
                SamplingParams(temperature=0.0, top_p=1.0, max_tokens=256, stop=["<|eot_id|>"])
            )
        except Exception as e:
            print(f"⚠️ [LLM Error] Skipping batch due to: {e}", flush=True)
            for entry in batch:
                results.append({
                    "Topic": entry["topic"],
                    "Action": "error",
                    "Explanation": str(e),
                    "Final_Path": str(entry["path"]),
                    "Placed": False,
                    "LLM_Calls": call_count.get(entry["topic"], 0),
                    "Total_Tokens": token_count.get(entry["topic"], 0)
                })
                completed.append(entry)
                if entry in active_topics:
                    active_topics.remove(entry)
            processed += len(batch)
            continue

        to_remove = []

        for i, output in enumerate(outputs):
            meta = metas[i]
            entry = meta["entry"]
            topic = entry["topic"]
            text = output.outputs[0].text.strip()

            try:
                parsed = extract_and_parse_json_blocks(text)
                if not parsed:
                    discard_topic(topic, "Malformed JSON")
                    results.append({
                        "Topic": topic,
                        "Action": "discard",
                        "Explanation": "Invalid JSON",
                        "Final_Path": str(entry["path"]),
                        "Placed": False
                    })
                    to_remove.append(entry)
                    continue

                prompt_tokens = len(tokenizer.encode(prompts[i]))
                response_tokens = len(tokenizer.encode(text))
                token_count[topic] = token_count.get(topic, 0) + prompt_tokens + response_tokens
                call_count[topic] = call_count.get(topic, 0) + 1

                decision = parsed[0]
                placed = apply_decision(topic, entry, meta["subnodes"], meta["is_leaf"], decision, taxonomy)

            except Exception as e:
                print(f"⚠️ [Topic Error] {topic}: {e}", flush=True)
                results.append({
                    "Topic": topic,
                    "Action": "error",
                    "Explanation": f"Exception during processing: {e}",
                    "Final_Path": str(entry["path"]),
                    "Placed": False,
                    "LLM_Calls": call_count.get(topic, 0),
                    "Total_Tokens": token_count.get(topic, 0)
                })
                to_remove.append(entry)

            if placed or entry["depth"] >= args.max_depth:
                to_remove.append(entry)

        for done in to_remove:
            if done in active_topics:
                active_topics.remove(done)
                completed.append(done)

                # Increment placed_total only if topic was truly placed
                last_result = next((r for r in reversed(results) if r["Topic"] == done["topic"]), None)
                if last_result and last_result["Placed"]:
                    placed_total += 1

        # # === Always log decisions for this batch ===
        # print("\n[BATCH SUMMARY]")
        # for meta in metas:
        #     entry = meta["entry"]
        #     topic = entry["topic"]
        #     result = next((r for r in reversed(results) if r["Topic"] == topic and r["Placed"]), None)
        #     if result:
        #         print(f"  📌 Topic: '{topic}' | Action: {result['Action']} | Node: {result.get('Node')} | Path: {result['Final_Path']} | Placed: True", flush=True)


        processed += len(to_remove)

        # === Checkpointing ===
        checkpoint_interval = 100
        current_checkpoint = (processed // checkpoint_interval) * checkpoint_interval
        previous_checkpoint = (processed_start // checkpoint_interval) * checkpoint_interval

        while current_checkpoint > previous_checkpoint:
            checkpoint_number = previous_checkpoint + checkpoint_interval
            ckpt_tax_path = f"{args.results_dir}/taxonomy/taxonomy_checkpoint_{checkpoint_number}.json"
            ckpt_excel_path = f"{args.results_dir}/excel/partial_results_{checkpoint_number}.xlsx"

            with open(ckpt_tax_path, "w") as f:
                json.dump(taxonomy, f, indent=2)
            pd.DataFrame(results).to_excel(ckpt_excel_path, index=False)
            print(f"[CHECKPOINT] {checkpoint_number} processed | Saved checkpoint.", flush=True)

            previous_checkpoint += checkpoint_interval

        # Update processed_start for next iteration
        processed_start = processed

# =============================
#            MAIN
# =============================

from pathlib import Path

def main():
    print("=== Starting Topic Placement ===",flush=True)
    
    # Load checkpointed taxonomy if available
    if args.load_checkpoint:
        checkpoint_taxonomy_file = f"{args.results_dir}/taxonomy/taxonomy_checkpoint_11200.json"
        if Path(checkpoint_taxonomy_file).exists():
            print(f"[INFO] Loading taxonomy from checkpoint: {checkpoint_taxonomy_file}", flush=True)
            with open(checkpoint_taxonomy_file, "r") as f:
                taxonomy = json.load(f)
        else:
            print(f"[INFO] No checkpoint found. Using fresh taxonomy.", flush=True)
            taxonomy = load_taxonomy_from_json(args.taxonomy_path)
    else:
        print(f"[INFO] Checkpoint flag is False. Using fresh taxonomy.", flush=True)
        taxonomy = load_taxonomy_from_json(args.taxonomy_path)
    
    llm, tokenizer = load_vllm_model()

    CHECKPOINT_INTERVAL = 100
    total_placed = 0

    # === Load previous results if available ===
    # Load previous results from checkpoint file if available
    if args.load_checkpoint:
        checkpoint_results_file = f"{args.results_dir}/excel/partial_results_11200.xlsx"
        if Path(checkpoint_results_file).exists():
            print(f"[INFO] Loading previous results from checkpoint: {checkpoint_results_file}", flush=True)
            previous_df = pd.read_excel(checkpoint_results_file)
            results.extend(previous_df.to_dict(orient="records"))
            processed_topics = set(previous_df["Topic"])
            total_processed = len(processed_topics)
            placed_topics = set(previous_df[previous_df["Placed"] == True]["Topic"])

            print(f"[INFO] Found {len(placed_topics)} placed | {total_processed} total topics processed.", flush=True)

    else:
        processed_topics = set()
        total_processed = 0




    # === Filter out topics that were already placed ===
    active_topics = [
    {"topic": t, "path": ["Science"], "depth": 0}
    for t in TOPICS if t not in processed_topics
]


    if not active_topics:
        print("✅ All topics have already been placed.",flush=True)
        return

    # === Multi-turn batched placement ===
    multi_turn_batched_placement(active_topics, taxonomy, llm, tokenizer, total_processed)

    # === Final save ===
    with open(f"{args.results_dir}/taxonomy/final_taxonomy_llm_incr.json", "w") as f:
        json.dump(taxonomy, f, indent=2)

    pd.DataFrame(results).to_excel(RESULTS_FILE, index=False)

    print("\n=== Completed ===",flush=True)
    print(f"Total Topics Placed: {len([r for r in results if r['Placed']])} / {len(TOPICS)}",flush=True)



if __name__ == "__main__":
    main()

# ============================= Example Usage =============================
# python /weka/scratch/dkhasha1/jash09/science-cartography-1/topic_clustering/llama/code/vllm/try_2/llama_2.py \
#     --topics_path /weka/scratch/dkhasha1/jash09/science-cartography-1/topic_clustering/unique_topics.txt \
#     --taxonomy_path /weka/scratch/dkhasha1/jash09/science-cartography-1/topic_clustering/taxonomies/seed/science_seed.json \
#     --results_dir /weka/scratch/dkhasha1/jash09/science-cartography-1/topic_clustering/llama/code/vllm/try_2 \
#     --batch_size 32 \
#     --max_depth 10 \
#     --max_tokens 32000 \
#     --max_response_tokens 256
# ========================================================================