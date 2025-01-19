import pandas as pd
import numpy as np
import pickle
import openai
from clustering_level1 import generate_embeddings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from clustering_87 import generate_cluster_name_and_summary


openai.api_key = 'YOUR_API_KEY'

# Load DataFrame and embeddings
df = pd.read_csv("approach1/results/cluster_87.csv")  # Replace with the actual file name
with open("approach1/embeddings/embeddings_87.pkl", "rb") as f:
    embeddings = pickle.load(f)

#group by cluster label and create a new dataframe with cluster label and cluster name summary joined
cluster_df = df.groupby('Cluster Label')['Cluster Name Summary'].apply('\n'.join).reset_index()

cluster_embeddings = generate_embeddings(cluster_df["Cluster Name Summary"].tolist())

print(f"Generated embeddings for {len(cluster_embeddings)} cluster summaries.")

# normalize the embeddings using StandardScaler
scaler = StandardScaler()
normalized_embeddings = scaler.fit_transform(cluster_embeddings)

# kmeans clustering with k=17 clusters
kmeans = KMeans(n_clusters=2, random_state=42)
cluster_labels = kmeans.fit_predict(normalized_embeddings)

# Add the cluster labels to the DataFrame
cluster_df["Cluster"] = cluster_labels

# Save the clustered DataFrame to a new CSV file
cluster_df.to_csv('approach1/results/cluster_17.csv', index=False)

print("kmeans clustering complete")

# create a new dataframe with cluster label and cluster name summary joined
cluster_df_17_names = df.groupby('Cluster Label')['Cluster Name Summary'].apply('\n'.join).reset_index()

#name the clusters using generate_cluster_name_and_summary function
cluster_names = []
cluster_summaries = []

for i, row in cluster_df_17_names.iterrows():
    cluster_title_summary = row["Cluster Name Summary"]
    cluster_name, cluster_summary = generate_cluster_name_and_summary(cluster_title_summary)
    cluster_names.append(cluster_name)
    cluster_summaries.append(cluster_summary)
    
cluster_df_17_names["Cluster Name"] = cluster_names
cluster_df_17_names["Cluster Summary"] = cluster_summaries

print("Cluster names and summaries generated.")
print(cluster_df_17_names["Cluster Summary"][0])
    
    
