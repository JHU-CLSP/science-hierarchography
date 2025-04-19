import json
import pandas as pd
import ast
import argparse
import os

def find_topic_path(taxonomy_dict, target_topic, path=None):
    """
    Recursively find the path of a topic in a nested taxonomy dictionary.
    """
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

def update_taxonomy(taxonomy_dict, topic_path, title, abstract, rationale):
    """
    Update taxonomy dictionary at the given topic path with a new paper entry.
    """
    current_level = taxonomy_dict
    for key in topic_path:
        current_level = current_level.setdefault(key, {})

    current_level.setdefault("Papers", [])

    paper_entry = {
        "Title": title.strip(),
        "Abstract": abstract.strip(),
        "Rationale": str(rationale).strip() if rationale else "No rationale provided"
    }

    if paper_entry not in current_level["Papers"]:
        current_level["Papers"].append(paper_entry)

def update_taxonomy_with_papers(csv_path, input_json_path, output_json_path, verbose=True):
    """
    Loads a taxonomy and CSV with mapped papers, then updates the taxonomy by embedding paper info under topics.
    """
    # Load CSV and taxonomy
    df = pd.read_csv(csv_path)
    with open(input_json_path, "r", encoding="utf-8") as json_file:
        taxonomy = json.load(json_file)

    skipped = 0
    unmatched = 0

    for index, row in df.iterrows():
        title = row.get("Title", "").strip()
        abstract = row.get("Abstract", "").strip()

        try:
            topics = ast.literal_eval(row["Topics"])
            rationales = ast.literal_eval(row["Rationales"])
        except (ValueError, SyntaxError):
            skipped += 1
            if verbose:
                print(f"⚠️ Skipping row {index} due to malformed Topics/Rationales.")
            continue

        for i, topic in enumerate(topics):
            rationale = rationales[i] if i < len(rationales) else "No rationale provided"
            topic_path = find_topic_path(taxonomy, topic)

            if topic_path:
                update_taxonomy(taxonomy, topic_path, title, abstract, rationale)
            else:
                unmatched += 1
                if verbose:
                    print(f"❌ Warning: Topic '{topic}' not found in taxonomy (row {index}).")

    with open(output_json_path, "w", encoding="utf-8") as output_file:
        json.dump(taxonomy, output_file, indent=4, ensure_ascii=False)

    if verbose:
        print("\n✅ Taxonomy updated with papers.")
        print(f"📄 Rows processed: {len(df)}")
        print(f"🚫 Rows skipped due to errors: {skipped}")
        print(f"❓ Topics not found in taxonomy: {unmatched}")
        print(f"📁 Output saved to: {output_json_path}\n")

# -----------------------------------
# CLI Entry Point
# -----------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add mapped papers into an existing taxonomy JSON.")
    parser.add_argument('--csv_path', type=str, required=True, help="Path to input CSV with Title, Abstract, Topics, Rationales")
    parser.add_argument('--taxonomy_path', type=str, required=True, help="Path to existing taxonomy JSON")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save updated taxonomy JSON with papers")
    parser.add_argument('--quiet', action='store_true', help="Suppress verbose output")

    args = parser.parse_args()
    update_taxonomy_with_papers(
        csv_path=args.csv_path,
        input_json_path=args.taxonomy_path,
        output_json_path=args.output_path,
        verbose=not args.quiet
    )

# Example usage:
# python map_papers.py \
#     --csv_path /path/to/final_outputs_updated.csv \
#     --taxonomy_path /path/to/merged_taxonomy_soc_sci.json \
#     --output_path /path/to/merged_taxonomy_soc_sci_w_papers.json
