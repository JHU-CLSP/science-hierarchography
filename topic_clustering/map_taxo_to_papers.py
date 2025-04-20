import json
import pandas as pd
import ast

# File paths
csv_file_path = "/Users/jashshah/Desktop/science-cartography-1/topic_clustering/taxonomies/results/final_outputs_updated.csv" 
json_file_path = "/Users/jashshah/Desktop/science-cartography-1/topic_clustering/taxonomies/exp/taxo_deepen/merged_taxonomy_soc_sci.json"
output_json_path = "/Users/jashshah/Desktop/science-cartography-1/topic_clustering/taxonomies/exp/taxo_deepen/merged_taxonomy_soc_sci_w_papers.json"

# Load CSV file
df = pd.read_csv(csv_file_path)

# Load JSON file
with open(json_file_path, "r", encoding="utf-8") as json_file:
    taxonomy = json.load(json_file)

# Function to recursively find the path of a topic in the taxonomy
def find_topic_path(taxonomy_dict, target_topic, path=None):
    if path is None:
        path = []
    for key, value in taxonomy_dict.items():
        if key == target_topic:
            return path + [key]
        elif isinstance(value, dict):
            new_path = find_topic_path(value, target_topic, path + [key])
            if new_path:
                return new_path
    return None

# Function to update the taxonomy and store multiple papers under the same topic
def update_taxonomy(taxonomy_dict, topic_path, title, abstract, rationale):
    current_level = taxonomy_dict
    for key in topic_path:
        if key not in current_level:
            current_level[key] = {}
        current_level = current_level[key]

    # Ensure a list of papers exists
    if "Papers" not in current_level:
        current_level["Papers"] = []

    # Ensure rationale is a valid string
    rationale = str(rationale).strip() if rationale else "No rationale provided"

    # Append the new paper entry instead of overwriting
    paper_entry = {
        "Title": title.strip(),
        "Abstract": abstract.strip(),
        "Rationale": rationale
    }

    if paper_entry not in current_level["Papers"]:  # Avoid duplicate entries
        current_level["Papers"].append(paper_entry)

# Iterate over each row in the CSV
for index, row in df.iterrows():
    title = row["Title"]
    abstract = row["Abstract"]

    # Convert Topics and Rationales columns from string representations to lists
    try:
        topics = ast.literal_eval(row["Topics"])  # Convert string list to actual list
        rationales = ast.literal_eval(row["Rationales"])  # Convert string list to actual list
    except (ValueError, SyntaxError):
        print(f"Skipping row {index} due to malformed Topics/Rationales.")
        continue  # Skip row if parsing fails

    # Ensure that each topic gets its corresponding rationale
    for i, topic in enumerate(topics):
        rationale = rationales[i] if i < len(rationales) else "No rationale provided"

        topic_path = find_topic_path(taxonomy, topic)
        if topic_path:
            update_taxonomy(taxonomy, topic_path, title, abstract, rationale)
        else:
            print(f"Warning: Topic '{topic}' not found in taxonomy.")

# Save the updated taxonomy
with open(output_json_path, "w", encoding="utf-8") as output_file:
    json.dump(taxonomy, output_file, indent=4, ensure_ascii=False)

print(f"Updated taxonomy saved to {output_json_path}")

