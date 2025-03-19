import openai
import pandas as pd
import json
import os
import re
import copy
import itertools
from dotenv import load_dotenv

load_dotenv()

# Set OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

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

def build_pruned_taxonomy(taxonomy, max_depth=2, current_depth=0):
    """
    Recursively build a pruned version of the taxonomy up to max_depth levels.
    Each node includes its "_id" so GPT can reference it.
    """
    if current_depth >= max_depth:
        return {}
    pruned = {}
    for node_name, node_value in taxonomy.items():
        if not isinstance(node_value, dict):
            continue
        pruned[node_name] = {"_id": node_value["_id"]}
        children = build_pruned_taxonomy(node_value, max_depth, current_depth + 1)
        for child_name, child_value in children.items():
            pruned[node_name][child_name] = child_value
    return pruned

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
    Remove the node with the given stable ID from the taxonomy and return a tuple:
    (original_key, removed_subtree). Returns (None, None) if not found.
    """
    parent, key, node = get_node_by_id(taxonomy, node_id)
    if parent is None or key is None:
        return None, None
    removed = parent.pop(key)
    return key, removed

def merge_dicts(dest, source):
    """
    Recursively merge two dictionaries.
    For keys present in both and that are dicts, merge them recursively;
    otherwise, overwrite dest with source.
    """
    for key, value in source.items():
        if key in dest and isinstance(dest[key], dict) and isinstance(value, dict):
            merge_dicts(dest[key], value)
        else:
            dest[key] = value
    return dest

def merge_nodes(taxonomy, source_id, target_id, merged_name=None):
    """
    Refined merge operation:
    1. Retrieve and make a deep copy of the source node (with its subtree).
    2. Remove the source node from the taxonomy.
    3. For each child (excluding "_id") in the source copy:
         - If the target node already has that child key, recursively merge the two subtrees.
         - Otherwise, add the child to the target node.
    4. Optionally, if merged_name is provided and is different from the current key,
         rename the target node in its parent's dictionary.
    """
    # Retrieve and remove the source node
    orig_key, source_node = remove_node_by_id(taxonomy, source_id)
    if source_node is None:
        print(f"[merge_nodes] Source node '{source_id}' not found.")
        return
    # Make a deep copy so that the full subtree is preserved
    source_copy = copy.deepcopy(source_node)
    
    # Locate the target node in the taxonomy
    parent_target, target_key, target_node = get_node_by_id(taxonomy, target_id)
    if target_node is None:
        print(f"[merge_nodes] Target node '{target_id}' not found.")
        return
    
    # For each child in the source copy, merge it into the target node
    for child_key, child_value in source_copy.items():
        if child_key == "_id":
            continue  # Skip the internal id
        if child_key in target_node:
            # If key exists, recursively merge the dictionaries
            merge_dicts(target_node[child_key], child_value)
        else:
            # Otherwise, simply add the new child
            target_node[child_key] = child_value
    
    # Optionally rename the target node if merged_name is provided
    if merged_name and merged_name != target_key:
        parent_target[merged_name] = parent_target.pop(target_key)
        print(f"Merged '{source_id}' into '{target_id}' and renamed target to '{merged_name}'.")
    else:
        print(f"Merged '{source_id}' into '{target_id}' (no rename).")

def move_node(taxonomy, node_id, new_parent_id):
    """
    Move a node (with its entire subtree) as follows:
      1. Retrieve a deep copy of the node while preserving its original key.
      2. Remove the original node from its current location.
      3. Attach the copied subtree to the new parent's subtree using the original key.
         If that key already exists in the new parent, merge the two nodes.
    """
    orig_key, node = remove_node_by_id(taxonomy, node_id)
    if node is None:
        print(f"[move_node] Node '{node_id}' not found.")
        return
    node_copy = copy.deepcopy(node)
    
    # Locate the new parent.
    _, _, new_parent = get_node_by_id(taxonomy, new_parent_id)
    if new_parent is None:
        print(f"[move_node] New parent '{new_parent_id}' not found.")
        return
    
    # If the original key already exists, merge the two subtrees.
    if orig_key in new_parent:
        print(f"Merging moved node '{node_id}' with existing node '{orig_key}' in new parent '{new_parent_id}'")
        merge_dicts(new_parent[orig_key], node_copy)
    else:
        new_parent[orig_key] = node_copy
        print(f"Moved node '{node_id}' under new parent '{new_parent_id}' as '{orig_key}'.")

def apply_change_log(taxonomy, change_log):
    """
    Apply a list of changes (only 'move' or 'merge') to the taxonomy.
    Each change is a dict that uses stable IDs.
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

class TaxonomyUpdater:
    def __init__(self, taxonomy, change_log):
        self.taxonomy = copy.deepcopy(taxonomy)
        self.change_log = change_log

    def update(self):
        return apply_change_log(self.taxonomy, self.change_log)

def extract_json_from_response(response_text):
    """
    Extract the JSON block from a GPT response that is enclosed in triple less-than and greater-than signs.
    This function looks for text between <<< and >>> (optionally with a "json" tag).
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
        print("No JSON block found in GPT response.")
        return None

def remove_ids(taxonomy):
    """
    Recursively remove the "_id" field from every node in the taxonomy.
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

if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # 1) LOAD TAXONOMIES
    # ---------------------------------------------------------------------
    pruned_taxo_path = "/Users/jashshah/Desktop/science-cartography-1/topic_clustering/taxonomies/results/pruned_taxo.json"
    bigger_taxo_path = "/Users/jashshah/Desktop/science-cartography-1/topic_clustering/taxonomies/results/merged_taxonomy.json"
    
    with open(pruned_taxo_path, "r") as f:
        pruned_taxonomy_raw = json.load(f)
    
    with open(bigger_taxo_path, "r") as f:
        bigger_taxonomy_raw = json.load(f)
    
    # ---------------------------------------------------------------------
    # 2) ASSIGN STABLE IDs TO THE BIGGER TAXONOMY (if not already present)
    # ---------------------------------------------------------------------
    bigger_taxonomy_with_ids = assign_ids(bigger_taxonomy_raw, prefix="node")
    
    # ---------------------------------------------------------------------
    # 3) BUILD (OR REBUILD) A PRUNED TAXONOMY UP TO N LEVELS
    # ---------------------------------------------------------------------
    pruned_taxonomy = build_pruned_taxonomy(bigger_taxonomy_with_ids, max_depth=2)
    
    # ---------------------------------------------------------------------
    # 4) ITERATIVE GPT PROMPTS AND UPDATES ON THE PRUNED TAXONOMY
    # ---------------------------------------------------------------------
    iterations_data = []
    current_view = pruned_taxonomy  # This is the view we feed into GPT.
    NUM_ITERATIONS = 20  # Adjust as needed.
    
    for iteration in range(NUM_ITERATIONS):
        prompt = f"""You are given the following pruned taxonomy subset with stable IDs:
{json.dumps(current_view, indent=4)}

Please suggest changes that involve ONLY moving or merging nodes.
For a move, provide:
- "action": "move"
- "node_id": the ID of the node to move
- "new_parent_id": the ID of the new parent

For a merge, provide:
- "action": "merge"
- "source_node_id": the ID of the node to merge from
- "target_node_id": the ID of the node to merge into
- Optionally, "merged_name": a new name for the merged node

Also include a brief "reason" for each change.

Return your suggestions in the following JSON format:
```json
{{
    "change_log": [
        {{
            "action": "move",         // or "merge"
            "node_id": "node_xxx",      // for move
            "new_parent_id": "node_yyy",// for move
            "source_node_id": "node_xxx",  // for merge
            "target_node_id": "node_yyy",  // for merge
            "merged_name": "Optional new name", // optional
            "reason": "brief justification"
        }}
    ]
}}
```
If no changes are needed, return an empty "change_log".
"""
        # Call GPT (replace with your actual API call)
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in taxonomy optimization. Only propose 'move' or 'merge' operations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        gpt_response_text = response["choices"][0]["message"]["content"]
        parsed_json = extract_json_from_response(gpt_response_text)
        if parsed_json is None:
            change_log = []
        else:
            change_log = parsed_json.get("change_log", [])
        
        # Apply the suggested changes to the current pruned view.
        updater = TaxonomyUpdater(current_view, change_log)
        updated_view = updater.update()
        
        iterations_data.append({
            "iteration": iteration + 1,
            "changes": change_log,
            "taxonomy_view": updated_view
        })
        
        current_view = updated_view
        print(f"Iteration {iteration+1} completed. Applied {len(change_log)} change(s).")
    
    # Save iteration details to Excel for record-keeping.
    df = pd.DataFrame(iterations_data)
    output_excel_path = "/Users/jashshah/Desktop/science-cartography-1/topic_clustering/taxonomies/results/final_taxonomy_iterations_stable_id.xlsx"
    df.to_excel(output_excel_path, index=False)
    print(f"Iteration details saved to {output_excel_path}")
    
    # ---------------------------------------------------------------------
    # 5) AGGREGATE ALL CHANGES AND APPLY TO THE FULL BIGGER TAXONOMY
    # ---------------------------------------------------------------------
    all_changes = []
    for it_data in iterations_data:
        all_changes.extend(it_data["changes"])
    
    final_bigger_taxonomy = apply_change_log(copy.deepcopy(bigger_taxonomy_with_ids), all_changes)
    
    # ---------------------------------------------------------------------
    # 6) SAVE TWO VERSIONS OF THE FINAL BIGGER TAXONOMY:
    #    a) WITH IDs
    #    b) WITHOUT IDs (after removing all "_id" keys)
    # ---------------------------------------------------------------------
    output_json_with_ids = "/Users/jashshah/Desktop/science-cartography-1/topic_clustering/taxonomies/results/final_bigger_taxonomy_with_ids.json"
    with open(output_json_with_ids, "w") as f:
        json.dump(final_bigger_taxonomy, f, indent=4)
    print(f"Final bigger taxonomy with IDs saved to {output_json_with_ids}")
    
    final_bigger_taxonomy_no_ids = remove_ids(copy.deepcopy(final_bigger_taxonomy))
    output_json_no_ids = "/Users/jashshah/Desktop/science-cartography-1/topic_clustering/taxonomies/results/final_bigger_taxonomy_no_ids.json"
    with open(output_json_no_ids, "w") as f:
        json.dump(final_bigger_taxonomy_no_ids, f, indent=4)
    print(f"Final bigger taxonomy without IDs saved to {output_json_no_ids}")