import pandas as pd
import json

def merge_dicts(a, b):
    """
    Recursively merges two dictionaries, combining nested structures.
    """
    merged = a.copy()
    for key in b:
        if key in merged:
            # Both values are dictionaries, so recurse
            if isinstance(merged[key], dict) and isinstance(b[key], dict):
                merged[key] = merge_dicts(merged[key], b[key])
            else:
                # Replace or handle non-dict values as needed
                merged[key] = b[key]
        else:
            merged[key] = b[key]
    return merged

# Read the Excel file
df = pd.read_excel('/Users/jashshah/Desktop/science-cartography-1/topic_clustering/50_topics_outputs/taxonomy_creation_outputs_50.xlsx')  # Update with your file path

# Initialize the merged taxonomy structure
merged_taxonomy = {}

# Iterate through each row to merge JSONs
for index, row in df.iterrows():
    json_str = row['implemented_taxonomy']
    try:
        current_dict = json.loads(json_str)
        merged_taxonomy = merge_dicts(merged_taxonomy, current_dict)
    except json.JSONDecodeError as e:
        print(f"Row {index + 1}: Invalid JSON - {e}")

# Convert the merged result to a formatted JSON string
final_json = json.dumps(merged_taxonomy, indent=4)

# Save or use the final JSON
print(final_json)  # Output to console
with open('/Users/jashshah/Desktop/science-cartography-1/topic_clustering/50_topics_outputs/50_topics_taxonomy.json', 'w') as f:
    f.write(final_json)