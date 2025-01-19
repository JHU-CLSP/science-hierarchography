import json
import os
import openai
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import re
import pickle
openai.api_key = 'YOUR_API_KEY'

# Function to generate embeddings using OpenAI API
def generate_embeddings(texts):
    embeddings = []
    for text in texts:
        try:
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"
            )
            embedding = response['data'][0]['embedding']
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error generating embedding for text: {text}\n{e}")
            embeddings.append([0] * 1536)  # Placeholder for errors
    return embeddings


#read csv file
summaries_df = pd.read_csv('approach1/results/summaries_topics.csv')
summaries = summaries_df["Summary"].tolist()
embeddings = generate_embeddings(summaries)

print(f"Generated embeddings for {len(embeddings)} summaries.")

# Normalize the embeddings using StandardScaler
scaler = StandardScaler()
normalized_embeddings = scaler.fit_transform(embeddings)

# store normalized embeddings as a dictionary in a pickle file for future use 
# key: summary, value: normalized embedding
normalized_embeddings_dict = dict(zip(summaries, normalized_embeddings))
# check if embeddings folder exists, if not create it
if not os.path.exists('approach1/embeddings'):
    os.makedirs('approach1/embeddings')
    
# save the normalized embeddings dictionary as pickle file
with open('approach1/embeddings/embeddings_summaries.pkl', 'wb') as f:
    pickle.dump(normalized_embeddings_dict, f)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=438, random_state=42)
cluster_labels = kmeans.fit_predict(normalized_embeddings)

print("kmeans clustering complete")

# Add the cluster labels to the DataFrame
summaries_df["Cluster"] = cluster_labels

# Save the clustered DataFrame to a new CSV file
summaries_df.to_csv('approach1/results/cluster_438.csv', index=False)

print('Clustering complete. Results saved to cluster_438.csv')

