import os
import subprocess

# Define base directory (adjust if needed)
BASE_DIR = "/Users/jashshah/Desktop/science-cartography-1/topic_clustering/"

# Define file paths for all intermediate steps
taxonomy_creation_script = os.path.join(BASE_DIR, "taxonomy_creation.py")
taxonomy_output_file = os.path.join(BASE_DIR, "50_topics_outputs/taxonomy_creation_outputs_50.xlsx")

json_formatter_script = os.path.join(BASE_DIR, "json_formatter.py")
formatted_taxonomy_file = os.path.join(BASE_DIR, "50_topics_outputs/taxonomy_creation_outputs_50.xlsx")

taxonomy_merge_script = os.path.join(BASE_DIR, "taxonomy_merge.py")
merged_taxonomy_file = os.path.join(BASE_DIR, "50_topics_outputs/50_topics_taxonomy.json")

map_taxo_to_papers_script = os.path.join(BASE_DIR, "map_taxo_to_papers.py")
papers_csv_file = os.path.join(BASE_DIR, "taxonomies/results/final_outputs_updated.csv")  # This file should exist
mapped_taxonomy_output = os.path.join(BASE_DIR, "50_topics_outputs/50_topics_taxonomy_w_papers.json")

# Ensure required files exist before proceeding
for required_file in [taxonomy_creation_script, json_formatter_script, taxonomy_merge_script, map_taxo_to_papers_script, papers_csv_file]:
    if not os.path.exists(required_file):
        raise FileNotFoundError(f"❌ Required file missing: {required_file}")

# Step 1: Generate Taxonomy
print("🔹 Step 1: Running taxonomy creation...")
subprocess.run(["python", taxonomy_creation_script], check=True)
print(f"✅ Taxonomy generated: {taxonomy_output_file}")

# Step 2: Format JSON Taxonomy
print("🔹 Step 2: Formatting taxonomy JSON...")
subprocess.run(["python", json_formatter_script], check=True)
print(f"✅ Formatted JSON saved: {formatted_taxonomy_file}")

# Step 3: Merge Taxonomies
print("🔹 Step 3: Merging taxonomy files...")
subprocess.run(["python", taxonomy_merge_script], check=True)
print(f"✅ Merged taxonomy saved: {merged_taxonomy_file}")

# Step 4: Map Papers to Taxonomy
print("🔹 Step 4: Mapping papers to taxonomy...")
subprocess.run(["python", map_taxo_to_papers_script], check=True)
print(f"✅ Final taxonomy with mapped papers saved: {mapped_taxonomy_output}")

print("🎯 Workflow completed successfully!")
