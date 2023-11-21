import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Generating random data points
np.random.seed(42)
mean = [0, 0]
cov = [[1, 0.8], [0.8, 1]]
data = np.random.multivariate_normal(mean, cov, 100)  # Generating 100 random 2D points

# Initializing PCA with 2 components
pca = PCA(n_components=2)

# Fitting the data to PCA
pca.fit(data)

# Transforming the data to its principal components
transformed_data = pca.transform(data)

# Plotting and saving the original and transformed data as images
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1])
plt.title('Original Data')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.savefig('original_data.png')  # Save the plot as original_data.png

plt.subplot(1, 2, 2)
plt.scatter(transformed_data[:, 0], transformed_data[:, 1])
plt.title('Transformed Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig('transformed_data.png')  # Save the plot as transformed_data.png

plt.tight_layout()

# If running in a local environment, display the plots
plt.show()
