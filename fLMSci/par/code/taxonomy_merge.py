import pandas as pd
import json
import argparse
import os

def merge_dicts(a, b):
    """
    Recursively merges two dictionaries, combining nested structures.
    """
    merged = a.copy()
    for key in b:
        if key in merged:
            if isinstance(merged[key], dict) and isinstance(b[key], dict):
                merged[key] = merge_dicts(merged[key], b[key])
            else:
                merged[key] = b[key]
        else:
            merged[key] = b[key]
    return merged

def merge_taxonomy_outputs(excel_path, json_output_path, verbose=True):
    """
    Reads taxonomy outputs from Excel, merges them into a single JSON.
    
    Parameters:
        excel_path (str): Path to the Excel file with 'implemented_taxonomy' column.
        json_output_path (str): Output path for the merged taxonomy JSON.
        verbose (bool): Whether to print progress to console.
    """
    df = pd.read_excel(excel_path)

    if 'implemented_taxonomy' not in df.columns:
        raise ValueError("Excel must contain a column named 'implemented_taxonomy'.")

    merged_taxonomy = {}
    failed_rows = 0

    for index, row in df.iterrows():
        json_str = row['implemented_taxonomy']

        if pd.isna(json_str) or not isinstance(json_str, str):
            failed_rows += 1
            continue

        try:
            current_dict = json.loads(json_str)
            merged_taxonomy = merge_dicts(merged_taxonomy, current_dict)
        except json.JSONDecodeError as e:
            failed_rows += 1
            if verbose:
                print(f"[Warning] Row {index + 1}: Invalid JSON - {e}")

    final_json_str = json.dumps(merged_taxonomy, indent=4)

    with open(json_output_path, 'w') as f:
        f.write(final_json_str)

    if verbose:
        print(f"\n✅ Merging complete.")
        print(f"📦 Total chunks merged: {len(df) - failed_rows}")
        print(f"❌ Rows failed to parse: {failed_rows}")
        print(f"📁 Merged taxonomy saved to: {json_output_path}\n")

# -------------------------------
# Command-line Interface
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge chunked taxonomy outputs from Excel to a final JSON.")
    parser.add_argument('--excel_path', type=str, required=True, help="Path to Excel file with taxonomy outputs")
    parser.add_argument('--json_output', type=str, required=True, help="Path to write merged JSON taxonomy")
    parser.add_argument('--quiet', action='store_true', help="Suppress console output")

    args = parser.parse_args()
    merge_taxonomy_outputs(
        excel_path=args.excel_path,
        json_output_path=args.json_output,
        verbose=not args.quiet
    )
    
# Example usage:
# python merge_taxonomy_outputs.py \
#     --excel_path taxonomy_creation_outputs.xlsx \
#     --json_output topics_taxonomy.json
