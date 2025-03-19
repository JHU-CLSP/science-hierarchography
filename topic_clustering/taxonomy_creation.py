import random
import openai
import json
import os
import pandas as pd
import re
import time
import concurrent.futures
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

# Read unique topics from the file
with open('/Users/jashshah/Desktop/science-cartography-1/topic_clustering/unique_topics.txt', 'r', encoding='utf-8') as file:
    unique_topics = file.readlines()

# Clean and deduplicate topics
unique_topics = [topic.strip() for topic in unique_topics]
unique_topics = list(set(unique_topics))

# Function to process topics in chunks
def chunk_topics(topics, chunk_size=50):
    """
    Generator to yield chunks of topics.
    :param topics: List of unique topics.
    :param chunk_size: Number of topics per chunk.
    """
    for i in range(0, len(topics), chunk_size):
        yield topics[i:i + chunk_size]

# Shuffle topics to add randomness
random.shuffle(unique_topics)

# Create chunks of topics
chunked_topics = list(chunk_topics(unique_topics, chunk_size=50))

# Function to extract all nodes from the taxonomy
def extract_all_nodes(taxonomy_dict):
    """
    Recursively extract all keys (nodes) from a nested dictionary.
    :param taxonomy_dict: A nested dictionary.
    :return: A set of all node values in the taxonomy.
    """
    nodes = set()
    
    def traverse_dict(d):
        for key, value in d.items():
            nodes.add(key)
            if isinstance(value, dict):
                traverse_dict(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        traverse_dict(item)
    
    traverse_dict(taxonomy_dict)
    return nodes

# Function to validate if all topics are included in the taxonomy
def validate_topics_in_taxonomy(topics, taxonomy_json_str):
    """
    Check which topics from the list remain missing in the taxonomy JSON.
    :param topics: List of topics to check.
    :param taxonomy_json_str: JSON string representing the taxonomy.
    :return: Tuple (matched_topics, missing_topics).
    """
    try:
        taxonomy_dict = json.loads(taxonomy_json_str)
        taxonomy_nodes = extract_all_nodes(taxonomy_dict)
        matched_topics = [topic for topic in topics if topic in taxonomy_nodes]
        missing_topics = [topic for topic in topics if topic not in taxonomy_nodes]
        return matched_topics, missing_topics
    except json.JSONDecodeError:
        print("Invalid JSON format in the taxonomy.")
        return [], topics

# Load seed taxonomy
with open('/Users/jashshah/Desktop/science-cartography-1/topic_clustering/taxonomies/seed/science_seed.json', 'r') as file:
    seed_taxonomy = json.load(file)

# Function to process a chunk of topics using GPT-4
def process_chunk(topics_chunk, seed_taxonomy, retries=3):
    for attempt in range(retries):
        try:
            prompt = (
                f'''You have a list of unique scientific topics and an initial seed taxonomy based on some scientific concepts. 
Task: Create a hierarchical taxonomy using the given topics. Make sure that the output has all the topics from the input list.
Instructions: 
1. Use the seed taxonomy as a starting point.
2. Do not change the current structure of the seed taxonomy and respect it. 
3. You are free to add new branches to accommodate new topics if they do not fit in the given seed taxonomy. Feel free to add subtrees but follow the primary structure. 
4. All topics need to appear only once and no topics that are not included in the input should be added. 
5. Do not create any new topics. Ensure children follow hyponymic relations. 
6. To place new terms make sure they follow is-a relationship and semantic meaning. 
Output format: JSON
Seed Taxonomy:
{seed_taxonomy}

Topics: {topics_chunk}
Output: Valid JSON format with the updated taxonomy structure.
'''
            )
            
            response = openai.ChatCompletion.create(
    model="gpt-4-1106-preview",
    messages=[
        {"role": "system", "content": "You are an experienced researcher who is skilled at reading scientific research and creating taxonomies based on scientific terms."},
        {"role": "user", "content": prompt},
    ],
    response_format = {"type": "json_object"},
    temperature=0.0
)
            
            resp = response['choices'][0]['message']['content']
            # json_match = re.search(r'```json(.*?)```', resp, re.DOTALL)
            #reply_content = json_match.group(1).strip() if json_match else resp
            
            if resp:
                matched_topics, missing_topics = validate_topics_in_taxonomy(topics_chunk, resp)
                return resp, missing_topics
        except Exception as e:
            print(f"Error processing chunk (Attempt {attempt + 1}): {e}")
            time.sleep(2)
    
    return None, topics_chunk

# Wrapper function that reprocesses missing topics until all are placed
def process_chunk_wrapper(chunk_id, topics_chunk, max_iterations=5):
    current_taxonomy = seed_taxonomy
    final_missing = None
    taxonomy = None
    for iteration in range(1, max_iterations + 1):
        print(f"Processing chunk id {chunk_id} - Iteration {iteration}")
        taxonomy, missing = process_chunk(topics_chunk, current_taxonomy)
        if taxonomy is None:
            print(f"Chunk id {chunk_id}: Failed to generate taxonomy.")
            break
        # Validate JSON format of the generated taxonomy
        try:
            loaded_taxonomy = json.loads(taxonomy)
        except json.JSONDecodeError:
            print(f"Chunk id {chunk_id}: Generated taxonomy JSON is invalid. Retrying iteration {iteration}...")
            continue
        # Validate which topics remain missing
        _, still_missing = validate_topics_in_taxonomy(topics_chunk, taxonomy)
        if not still_missing:
            print(f"Chunk id {chunk_id}: All topics placed.")
            return {
                "chunk_id": chunk_id,
                "chunked_topics": topics_chunk,
                "implemented_taxonomy": taxonomy,
                "missing_topics": []
            }
        else:
            print(f"Chunk id {chunk_id}: {len(still_missing)} topics still missing; reprocessing.")
            current_taxonomy = loaded_taxonomy
            final_missing = still_missing
    return {
        "chunk_id": chunk_id,
        "chunked_topics": topics_chunk,
        "implemented_taxonomy": taxonomy,
        "missing_topics": final_missing if final_missing is not None else topics_chunk
    }

# Process chunks in parallel using ThreadPoolExecutor
results_list = []
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    future_to_chunk = {
        executor.submit(process_chunk_wrapper, idx, chunk): idx 
        for idx, chunk in enumerate(chunked_topics, start=1)
    }
    for future in concurrent.futures.as_completed(future_to_chunk):
        result = future.result()
        results_list.append(result)
        print(f"Completed processing chunk id: {result['chunk_id']}.")

# Convert results to DataFrame
df = pd.DataFrame(results_list)

# Save to Excel file
output_excel_path = "/Users/jashshah/Desktop/science-cartography-1/topic_clustering/50_topics_outputs/taxonomy_creation_outputs_50.xlsx"
df.to_excel(output_excel_path, index=False)

print(f"Processing complete. Results saved to '{output_excel_path}'.")