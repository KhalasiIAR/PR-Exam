import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate some random data for demonstration purposes
np.random.seed(1)
data, _ = make_blobs(n_samples=200, centers=2, cluster_std=1.5)

# Specify the number of clusters (K)
K = 2

# Apply K-means clustering
kmeans = KMeans(n_clusters=K)
kmeans.fit(data)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plot the results
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=10)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=100, label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
