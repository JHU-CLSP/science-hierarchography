import pandas as pd
import numpy as np
import pickle
import openai
from clustering_level1 import generate_embeddings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()

# Get the API key
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

# Load DataFrame and embeddings
df = pd.read_csv("approach1/results/cluster_438.csv")  # Replace with the actual file name
with open("approach1/embeddings/embeddings_summaries.pkl", "rb") as f:
    embeddings = pickle.load(f)

# Function to compute the centroid of a cluster
def compute_centroid(cluster_points):
    cluster_embeddings = np.array([embeddings[summary] for summary in cluster_points if summary in embeddings])
    if len(cluster_embeddings) > 0:
        return np.mean(cluster_embeddings, axis=0)
    return None

# Function to find the top 10 closest points to the centroid
def find_closest_points(cluster_points, centroid):
    distances = []
    for summary in cluster_points:
        if summary in embeddings:
            distance = np.linalg.norm(embeddings[summary] - centroid)
            distances.append((summary, distance))
    distances = sorted(distances, key=lambda x: x[1])
    return [x[0] for x in distances[:10]]  # Top 10 closest points

# Create a dictionary to store the result
cluster_results = []

for cluster_number, group in df.groupby("Cluster"):
    summaries = group["Summary"].tolist()
    titles = group["Title"].tolist()
    cluster_points = group["Summary"].tolist()
    
    # Compute centroid
    centroid = compute_centroid(cluster_points)
    if centroid is None:
        continue
    
    # Find top 10 closest points
    closest_points = find_closest_points(cluster_points, centroid)
    
    # Extract titles and summaries for the top 10 points
    top_10_titles = [titles[i] for i, summary in enumerate(summaries) if summary in closest_points]
    top_10_summaries = [summaries[i] for i, summary in enumerate(summaries) if summary in closest_points]
    
    # Combine titles and summaries into formatted pairs
    top_10_pairs = [
        f"Title: {title}\nSummary: {summary}\n" for title, summary in zip(top_10_titles, top_10_summaries)
    ]
    
    # Append results
    cluster_results.append({
        "cluster_number": cluster_number,
        "top_10_title_summary": "\n".join(top_10_pairs),  # Original formatted pairs
        "top_10_titles": top_10_titles,                  # Separate titles
        "top_10_summaries": top_10_summaries             # Separate summaries
    })

# Create the final DataFrame
df438_w_centroids = pd.DataFrame(cluster_results)
df438_w_centroids.to_csv("approach1/results/cluster_438_with_centroids.csv", index=False)

def generate_cluster_name_and_summary(summaries, user_prompt):
    """
    Generate a cluster name and summary using OpenAI API.
    
    Args:
    - summaries (list): A list of formatted titles and summaries for the cluster.
    
    Returns:
    - tuple: (cluster_name, cluster_summary)
    """
    
    messages = [
        {
            "role": "system",
            "content": "You are a scientist tasked with naming clusters of research papers based on their titles and summaries."
        },
        {
            "role": "user",
            "content": f"""
Here are the titles and summaries for a cluster:

{summaries}

{user_prompt}
"""
        }
    ]
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.0,
        )
        
        output = response['choices'][0]['message']['content']
        
        start = output.find("{")
        end = output.rfind("}") + 1
        json_response = output[start:end]
        
        cluster_data = eval(json_response)
        cluster_name = cluster_data.get("Cluster Name", None)
        cluster_summary = cluster_data.get("Cluster Summary", None)
        
        return cluster_name, cluster_summary
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return None, None
    

# Generate cluster names and summaries and add them to the DataFrame
cluster_names = []
cluster_summaries = []

with open('approach1/prompts/cluster_naming.txt', 'r', encoding='utf-8') as file:
    user_prompt = file.read().strip()


for i, row in df438_w_centroids.iterrows():
    cluster_title_summary = row["top_10_title_summary"]
    cluster_name, cluster_summary = generate_cluster_name_and_summary(cluster_title_summary, user_prompt)
    cluster_names.append(cluster_name)
    cluster_summaries.append(cluster_summary)
    
df438_w_centroids["Cluster Name"] = cluster_names

df438_w_centroids["Cluster Summary"] = cluster_summaries

df438_w_centroids.to_csv("approach1/results/cluster_438_with_names.csv", index=False)
    
print("Cluster names and summaries generated and saved.")

#create a new column to save a combination of cluster name and cluster summary with a separator "\n"

df438_w_centroids["Cluster Name Summary"] = df438_w_centroids["Cluster Name"] + "\n" + df438_w_centroids["Cluster Summary"]

# generate embeddings for the cluster name summary column and save it as a pickle file
cluster_name_summaries = df438_w_centroids["Cluster Name Summary"].tolist()
cluster_name_summaries_embeddings = generate_embeddings(cluster_name_summaries)

# Normalize the embeddings using StandardScaler
scaler = StandardScaler()
normalized_cluster_name_summaries_embeddings = scaler.fit_transform(cluster_name_summaries_embeddings)

# store normalized embeddings as a dictionary in a pickle file for future use
# key: cluster name summary, value: normalized embedding
normalized_cluster_name_summaries_embeddings_dict = dict(zip(cluster_name_summaries, normalized_cluster_name_summaries_embeddings))

# save the normalized embeddings dictionary as pickle file
with open('approach1/embeddings/embeddings_438.pkl', 'wb') as f:
    pickle.dump(normalized_cluster_name_summaries_embeddings_dict, f)
    
print("Cluster name and summary embeddings generated and saved for level 438.")

# Perform KMeans clustering on the cluster name and summary embeddings
kmeans = KMeans(n_clusters=87, random_state=42)
cluster_labels = kmeans.fit_predict(normalized_cluster_name_summaries_embeddings)

print("kmeans clustering complete")

# Add the cluster labels to the DataFrame
df438_w_centroids["Cluster Label"] = cluster_labels

# Save the clustered DataFrame to a new CSV file
df438_w_centroids.to_csv('approach1/results/cluster_87.csv', index=False)


    
