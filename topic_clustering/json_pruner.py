import json

def truncate_json(data, max_level, current_level=1):
    """
    Truncates a nested JSON dictionary to the specified maximum level.
    
    Args:
        data (dict): The JSON data to truncate.
        max_level (int): The maximum depth level to retain (1-based).
        current_level (int): The current depth level (used recursively).
    
    Returns:
        dict: The truncated JSON data.
    """
    if current_level > max_level:
        return {}
    if not isinstance(data, dict):
        return data
    truncated = {}
    for key, value in data.items():
        if current_level == max_level:
            truncated[key] = {}
        else:
            truncated[key] = truncate_json(value, max_level, current_level + 1)
    return truncated

# Example usage:
if __name__ == "__main__":
    # Load the JSON data from a file
    with open('/Users/jashshah/Desktop/science-cartography-1/topic_clustering/taxonomies/results/merged_taxonomy_updated.json', 'r') as f:
        original_data = json.load(f)
    
    # Specify the desired level (1-based)
    desired_level = 4  # Change this value to truncate at different levels
    
    # Truncate the data
    truncated_data = truncate_json(original_data, desired_level)
    
    # Save the truncated data to a new file
    with open('/Users/jashshah/Desktop/science-cartography-1/topic_clustering/taxonomies/results/pruned_taxo_2.json', 'w') as f:
        json.dump(truncated_data, f, indent=4)
    
    print(f"Truncated JSON up to level {desired_level} has been saved to output.json")