import openai
import pandas as pd
import json
import copy
import re
import itertools
import os
from dotenv import load_dotenv
load_dotenv()

# Set OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

##################################################
# (1) Utility Functions (Assigning IDs, Searching, etc.)
##################################################

def assign_ids(taxonomy, prefix="node", counter=None):
    """
    Recursively assign stable IDs to each node in a nested taxonomy.
    Each node (a dict) gets an "_id" field if not already present.
    """
    if counter is None:
        counter = itertools.count(1)
    for node_name, node_value in list(taxonomy.items()):
        if not isinstance(node_value, dict):
            continue
        if "_id" not in node_value:
            node_value["_id"] = f"{prefix}_{next(counter)}"
        assign_ids(node_value, prefix=prefix, counter=counter)
    return taxonomy

def get_node_by_id(taxonomy, node_id):
    """
    Recursively search for a node (a dict) with the given "_id".
    Returns a tuple (parent_dict, key, node_dict) if found; else (None, None, None).
    """
    for key, value in list(taxonomy.items()):
        if not isinstance(value, dict):
            continue
        if value.get("_id") == node_id:
            return (taxonomy, key, value)
        parent, found_key, found_node = get_node_by_id(value, node_id)
        if parent is not None:
            return (parent, found_key, found_node)
    return (None, None, None)

def remove_node_by_id(taxonomy, node_id):
    """
    Remove the node with the given stable ID from the taxonomy.
    Returns (original_key, removed_subtree). (None, None) if not found.
    """
    parent, key, node = get_node_by_id(taxonomy, node_id)
    if parent is None or key is None:
        return None, None
    removed = parent.pop(key)
    return key, removed

def merge_dicts(dest, source):
    """
    Recursively merge source into dest. 
    If dest[key] and source[key] are dicts, merges them deeply.
    Otherwise, overwrites dest[key] with source[key].
    """
    for key, value in source.items():
        if key in dest and isinstance(dest[key], dict) and isinstance(value, dict):
            merge_dicts(dest[key], value)
        else:
            dest[key] = value
    return dest

def remove_ids(taxonomy):
    """
    Recursively remove "_id" fields from a taxonomy dictionary.
    """
    if isinstance(taxonomy, dict):
        new_tax = {}
        for key, value in taxonomy.items():
            if key == "_id":
                continue
            new_tax[key] = remove_ids(value)
        return new_tax
    elif isinstance(taxonomy, list):
        return [remove_ids(item) for item in taxonomy]
    else:
        return taxonomy

##################################################
# (2) Taxonomy Move & Merge Operations
##################################################

def move_node(taxonomy, node_id, new_parent_id):
    """
    Move an entire node (subtree) from its current parent to a new parent.
    If the new parent already has a child with the same key, merges them.
    """
    orig_key, node = remove_node_by_id(taxonomy, node_id)
    if node is None:
        print(f"[move_node] Node '{node_id}' not found.")
        return
    node_copy = copy.deepcopy(node)
    
    # Locate the new parent
    _, _, new_parent = get_node_by_id(taxonomy, new_parent_id)
    if new_parent is None:
        print(f"[move_node] New parent '{new_parent_id}' not found.")
        return
    
    # If that key already exists under new_parent, recursively merge
    if orig_key in new_parent:
        print(f"Merging moved node '{node_id}' with existing node '{orig_key}' in new parent '{new_parent_id}'")
        merge_dicts(new_parent[orig_key], node_copy)
    else:
        new_parent[orig_key] = node_copy
        print(f"Moved node '{node_id}' under '{new_parent_id}' as '{orig_key}'.")

def merge_nodes(taxonomy, source_id, target_id, merged_name=None):
    """
    Merge 'source_id' subtree into 'target_id' subtree.
    If merged_name is supplied, the target node is renamed to merged_name in the parent's dict.
    """
    orig_key, source_node = remove_node_by_id(taxonomy, source_id)
    if source_node is None:
        print(f"[merge_nodes] Source node '{source_id}' not found.")
        return
    source_copy = copy.deepcopy(source_node)
    
    parent_target, target_key, target_node = get_node_by_id(taxonomy, target_id)
    if target_node is None:
        print(f"[merge_nodes] Target node '{target_id}' not found.")
        return
    
    # Merge children of source into target
    for child_key, child_value in source_copy.items():
        if child_key == "_id":
            continue
        if child_key in target_node:
            merge_dicts(target_node[child_key], child_value)
        else:
            target_node[child_key] = child_value
    
    # Optionally rename the merged target
    if merged_name and merged_name != target_key:
        parent_target[merged_name] = parent_target.pop(target_key)
        print(f"Merged '{source_id}' into '{target_id}' and renamed target to '{merged_name}'.")
    else:
        print(f"Merged '{source_id}' into '{target_id}' (no rename).")

def apply_change_log(taxonomy, change_log):
    """
    Apply a list of 'move' or 'merge' operations (with optional rename).
    """
    for change in change_log:
        action = change.get("action")
        if action == "move":
            node_id = change.get("node_id")
            new_parent_id = change.get("new_parent_id")
            move_node(taxonomy, node_id, new_parent_id)
        elif action == "merge":
            source_id = change.get("source_node_id")
            target_id = change.get("target_node_id")
            merged_name = change.get("merged_name")
            merge_nodes(taxonomy, source_id, target_id, merged_name)
        else:
            print(f"[apply_change_log] Unknown action '{action}', skipping.")
    return taxonomy

##################################################
# (3) Building a Pruned Subtree (k=2)
##################################################

def build_subtree_pruned(taxonomy, node_id, max_depth=2, current_depth=0):
    """
    Returns a pruned subtree (up to `max_depth` levels) starting at the node 
    with stable ID 'node_id'.
    """
    parent, key, node_dict = get_node_by_id(taxonomy, node_id)
    if node_dict is None:
        return {}
    
    # Build a top-level structure: { node_name: { "_id": node_id, ... } }
    pruned = {
        key: {
            "_id": node_dict["_id"]
        }
    }
    
    # If we haven't reached max depth, recurse for each child
    if current_depth < max_depth - 1:
        for child_name, child_val in node_dict.items():
            if child_name == "_id" or not isinstance(child_val, dict):
                continue
            child_id = child_val["_id"]
            child_subtree = build_subtree_pruned(taxonomy, child_id, max_depth, current_depth + 1)
            for ckey, cval in child_subtree.items():
                pruned[key][ckey] = cval
    
    return pruned

##################################################
# (4) GPT Response Parsing (Using <<< and >>>)
##################################################

def extract_json_from_response(response_text):
    """
    Extract the JSON block from a GPT response enclosed in <<< ... >>>.
    We'll look for something like:
        <<<json
        {
            ...
        }
        >>>
    """
    pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    if matches:
        json_text = matches[0]
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return None
    else:
        print("No <<< ... >>> block found in GPT response.")
        return None

##################################################
# (5) Recursive Processing with k=2
##################################################

# We'll maintain a global or external "iteration" counter 
# to log how many GPT calls have been made so far.
class GlobalIterationCounter:
    def __init__(self):
        self.count = 0
    
    def increment(self):
        self.count += 1
        return self.count

def process_node_recursively(
    taxonomy,
    node_id,
    level=0,
    max_depth=2,
    visited=None,
    openai_api_key=None,
    gpt_call_log=None,
    global_iter_counter=None
):
    """
    1) Build a pruned (k=2) view of this node & children.
    2) Prompt GPT for move/merge changes (expecting <<< ... >>> JSON).
    3) Apply those changes to the *master* taxonomy.
    4) Recursively process each child if within max_depth.
    5) Log extended data (iteration, node_id, subtree states, prompt, response, etc.).
    """
    if visited is None:
        visited = set()
    if global_iter_counter is None:
        global_iter_counter = GlobalIterationCounter()
    
    if node_id in visited:
        return  # Avoid re-processing the same node if merges reintroduce it
    visited.add(node_id)
    
    # Build the pruned subtree (before GPT modifies anything)
    pruned_view_before = build_subtree_pruned(taxonomy, node_id, max_depth=max_depth)
    if not pruned_view_before:
        return
    
    # Create GPT prompt, instructing it to return the JSON in <<< ... >>> fences
    user_prompt = f"""
You are given the following pruned taxonomy subset (up to depth=2) with stable IDs:

{json.dumps(pruned_view_before, indent=4)}

Please suggest changes that involve ONLY moving or merging nodes.

For a move:
  "action": "move"
  "node_id": the ID of the node to move
  "new_parent_id": the ID of the new parent

For a merge:
  "action": "merge"
  "source_node_id": the ID of the node to merge from
  "target_node_id": the ID of the node to merge into
  "merged_name": (optional new name)

Also include a brief "reason" for each change.

Return your suggestions in the following JSON format, enclosed in triple angle brackets:

```json
{{
  "change_log": [
    {{
      "action": "...",
      "node_id": "...",
      "new_parent_id": "...",
      "source_node_id": "...",
      "target_node_id": "...",
      "merged_name": "...",
      "reason": "..."
    }}
  ]
}}
```

If no changes are needed, return an empty "change_log".
"""
    # Make GPT call
    if openai_api_key:
        openai.api_key = openai_api_key
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in taxonomy optimization. Only propose 'move' or 'merge' operations."
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.0
        )
    except Exception as e:
        print(f"Error in GPT call: {e}")
        # store the prompt in a text file as it may be too long for the console
        with open("/Users/jashshah/Desktop/science-cartography-1/topic_clustering/taxonomies/exp/taxo_deepen/gpt_prompt.txt", "w") as f:
            f.write(user_prompt)
    gpt_response_text = response["choices"][0]["message"]["content"]
    
    # Increment iteration count
    iteration_num = global_iter_counter.increment()

    # Parse out JSON from the <<< ... >>> block
    parsed_json = extract_json_from_response(gpt_response_text)
    if parsed_json is None:
        change_log = []
    else:
        change_log = parsed_json.get("change_log", [])
    
    # Apply changes
    apply_change_log(taxonomy, change_log)
    
    # Build a pruned subtree (after GPT changes) for logging
    # (Re-find this node in case merges or moves changed its parent)
    parent, new_key, new_node_dict = get_node_by_id(taxonomy, node_id)
    pruned_view_after = {}
    if new_node_dict:
        pruned_view_after = build_subtree_pruned(taxonomy, node_id, max_depth=max_depth)
    
    # Optionally log GPT call info with extended data
    if gpt_call_log is not None:
        gpt_call_log.append({
            "iteration": iteration_num,
            "node_id": node_id,
            "recursion_level": level,
            "pruned_view_before": json.dumps(pruned_view_before, indent=2),
            "prompt": user_prompt,
            "raw_gpt_response": gpt_response_text,
            "parsed_change_log": change_log,
            "pruned_view_after": json.dumps(pruned_view_after, indent=2)
        })
    
    # If the node was merged away or removed, we stop
    if new_node_dict is None:
        return
    
    # If not at the max depth, go deeper into each child
    if level < max_depth - 1:
        for child_name, child_val in new_node_dict.items():
            if child_name == "_id" or not isinstance(child_val, dict):
                continue
            child_id = child_val["_id"]
            process_node_recursively(
                taxonomy,
                child_id,
                level+1,
                max_depth,
                visited,
                openai_api_key,
                gpt_call_log,
                global_iter_counter
            )

##################################################
# (6) Example Main Driver
##################################################

if __name__ == "__main__":
    # Example usage: set or read an env var for OPENAI_API_KEY
    # os.environ["OPENAI_API_KEY"] = "sk-..."
    # openai_api_key = os.getenv("OPENAI_API_KEY")
    
    with open('/Users/jashshah/Desktop/science-cartography-1/topic_clustering/taxonomies/results/final_bigger_taxonomy_no_ids.json', 'r') as f:
        example_taxonomy = json.load(f)
    
    # If your taxonomy does NOT have stable IDs yet, uncomment:
    example_taxonomy = assign_ids(example_taxonomy)

    # A list to store logs of each GPT call (with more data)
    gpt_call_log = []

    # We'll start from the root node "node_1"
    root_node_id = "node_1"

    # Recursively process the entire taxonomy in chunks (k=2)
    process_node_recursively(
        taxonomy=example_taxonomy,
        node_id=root_node_id,
        level=0,
        max_depth=2,
        visited=None,
        openai_api_key=None,   # or your actual key
        gpt_call_log=gpt_call_log,
        global_iter_counter=GlobalIterationCounter()
    )

    # After recursion, example_taxonomy is updated with GPT suggestions.
    with open("/Users/jashshah/Desktop/science-cartography-1/topic_clustering/taxonomies/exp/taxo_deepen/final_taxonomy.json", "w") as f:
        json.dump(example_taxonomy, f, indent=4)
    print("Done. Final taxonomy saved to 'final_taxonomy.json'.")

    # Convert GPT call logs to a DataFrame and save to Excel
    df_calls = pd.DataFrame(gpt_call_log)
    df_calls.to_excel("/Users/jashshah/Desktop/science-cartography-1/topic_clustering/taxonomies/exp/taxo_deepen/gpt_call_log.xlsx", index=False)
    print("GPT call log saved to 'gpt_call_log.xlsx'.")