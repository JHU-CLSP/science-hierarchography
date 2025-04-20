"""
clustering.py: KMeans-based clustering with scikit-learn.
"""

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

class KMeansClustering:
    def __init__(self, random_state=42):
        self.random_state = random_state

    def perform_clustering(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        scaler = StandardScaler()
        X = scaler.fit_transform(embeddings)
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        labels = kmeans.fit_predict(X)
        return labels

class SpectralClust:
    def __init__(self, random_state=42):
        self.random_state = random_state
    
    def perform_clustering(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        scaler = StandardScaler()
        X = scaler.fit_transform(embeddings)
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            random_state=self.random_state,
            affinity='rbf',
            n_init=10
        )
        labels = spectral.fit_predict(X)
        return labels

class AgglomerativeClust:
    def __init__(self, random_state=42):
        self.random_state = random_state
    
    def perform_clustering(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        scaler = StandardScaler()
        X = scaler.fit_transform(embeddings)
        agglomerative = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )
        labels = agglomerative.fit_predict(X)
        return labels
