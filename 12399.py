# Pattern Recognition :

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate some random data for demonstration purposes
np.random.seed(42)
data = np.random.rand(100, 2)

# Define the number of clusters (K)
k = 3

# Instantiate the KMeans model
kmeans = KMeans(n_clusters=k)

# Fit the model to the data
kmeans.fit(data)

# Get the cluster assignments and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plot the data points and centroids
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
