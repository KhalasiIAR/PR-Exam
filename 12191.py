# ImporƟng necessary libraries
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs 
# GeneraƟng sample data
n_samples = 300 
n_features = 2 
n_clusters = 3 
X, y = make_blobs(n_samples=n_samples, n_features=n_features, 
centers=n_clusters, random_state=42) 
# Applying K-means clustering 
kmeans = KMeans(n_clusters=n_clusters) 
kmeans.fit(X) 
y_kmeans = kmeans.predict(X) 
centers = kmeans.cluster_centers_ 
# Ploƫng the clustered data points and centroids
plt.figure(figsize=(8, 6)) 
plt.scaƩer(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', s=50, alpha=0.7)
plt.scaƩer(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, 
label='Centroids') 
plt.xlabel('Feature 1') 
plt.ylabel('Feature 2')
plt.Ɵtle('K-means Clustering')
plt.legend()
plt.show()
