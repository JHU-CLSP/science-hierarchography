import pandas as pd
import json

# Load the Excel file
file_path = "/Users/jashshah/Desktop/science-cartography-1/topic_clustering/50_topics_outputs/taxonomy_creation_outputs_50.xlsx"  # Replace with your file path
df = pd.read_excel(file_path)

# Specify the column containing JSON strings
json_column = "implemented_taxonomy"  # Replace with the actual column name

# Function to format JSON strings
def format_json(json_str):
    try:
        parsed = json.loads(json_str)  # Parse JSON string
        return json.dumps(parsed, indent=4)  # Format JSON with indentation
    except json.JSONDecodeError:
        return json_str  # Return original string if it's not valid JSON

# Apply formatting to each row in the column and replace single quotes with double quotes

#df[json_column] = df[json_column].str.replace("'", '"')
df[json_column] = df[json_column].astype(str).apply(format_json)

# Save the formatted data back to a new Excel file
output_path = "/Users/jashshah/Desktop/science-cartography-1/topic_clustering/50_topics_outputs/taxonomy_creation_outputs_50.xlsx"  # Replace with desired output path
df.to_excel(output_path, index=False)

print(f"Formatted JSON saved to {output_path}")