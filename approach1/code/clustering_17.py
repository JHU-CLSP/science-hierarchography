import pandas as pd
import numpy as np
import pickle
import openai
from clustering_level1 import generate_embeddings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from clustering_87 import generate_cluster_name_and_summary
import os
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()

# Get the API key
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

# Load DataFrame and embeddings
df = pd.read_csv("approach1/results/cluster_87.csv")  # Replace with the actual file name
df.dropna(inplace=True)
#group by cluster label and create a new dataframe with cluster label and cluster name summary joined
cluster_df = df.groupby('Cluster Label')['Cluster Name Summary'].apply('\n'.join).reset_index()

print("Grouped clusters by label.")

with open('approach1/prompts/cluster_naming.txt', 'r', encoding='utf-8') as file:
    user_prompt = file.read().strip()

#generate cluster names and summaries using generate_cluster_name_and_summary function
cluster_names_87 = []
cluster_summaries_87 = []

for i, row in cluster_df.iterrows():
    cluster_title_summary = row["Cluster Name Summary"]
    cluster_name, cluster_summary = generate_cluster_name_and_summary(cluster_title_summary, user_prompt)
    cluster_names_87.append(cluster_name)
    cluster_summaries_87.append(cluster_summary)
    
cluster_df["Cluster Name"] = cluster_names_87
cluster_df["Cluster Summary"] = cluster_summaries_87

print("Cluster names and summaries generated.")

# Create a new column as combination of cluster name and cluster summary
# Title: Cluster Name \n Summary: Cluster Summary
cluster_df["Cluster Description"] = "Title: " + cluster_df["Cluster Name"] + "\nSummary: " + cluster_df["Cluster Summary"]
cluster_df.dropna(inplace=True)
cluster_df.to_csv('approach1/results/cluster_87_with_names.csv', index=False)

cluster_embeddings = generate_embeddings(cluster_df["Cluster Description"].tolist())
print(f"Generated embeddings for {len(cluster_embeddings)} cluster summaries.")

# normalize the embeddings using StandardScaler
scaler = StandardScaler()
normalized_embeddings = scaler.fit_transform(cluster_embeddings)

# kmeans clustering with k=17 clusters
kmeans = KMeans(n_clusters=17, random_state=42)
cluster_labels = kmeans.fit_predict(normalized_embeddings)

# Add the cluster labels to the DataFrame
cluster_df["Cluster"] = cluster_labels

# Save the clustered DataFrame to a new CSV file
cluster_df.to_csv('approach1/results/cluster_17.csv', index=False)

print("kmeans clustering complete")
cluster_df = pd.read_csv("approach1/results/cluster_17.csv")  # Replace with the actual file name
# create a new dataframe with cluster label and cluster name summary joined
cluster_df_17_names = cluster_df.groupby('Cluster')['Cluster Description'].apply('\n'.join).reset_index()

#name the clusters using generate_cluster_name_and_summary function
cluster_names = []
cluster_summaries = []

for i, row in cluster_df_17_names.iterrows():
    cluster_title_summary = row["Cluster Description"]
    cluster_name, cluster_summary = generate_cluster_name_and_summary(cluster_title_summary, user_prompt)
    cluster_names.append(cluster_name)
    cluster_summaries.append(cluster_summary)
    
cluster_df_17_names["Cluster Name"] = cluster_names
cluster_df_17_names["Cluster Summary"] = cluster_summaries

cluster_df_17_names["Cluster Description Final"] = "Title: " + cluster_df_17_names["Cluster Name"] + "\nSummary: " + cluster_df_17_names["Cluster Summary"]

cluster_df_17_names.to_csv('approach1/results/cluster_17_with_names.csv', index=False)

print("Cluster names and summaries generated for 17 clusters.")
    
    
