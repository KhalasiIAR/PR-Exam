import matplotlib.pyplot as plt

x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

# Scatter plot
plt.scatter(x, y)
plt.title('Scatter Plot')
plt.xlabel('X')
plt.ylabel('Y')
scatter_filename = 'scatter_plot.png'
plt.savefig(scatter_filename)


from sklearn.cluster import KMeans

data = list(zip(x, y))
inertias = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

# Elbow method plot
plt.plot(range(1, 11), inertias, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
elbow_filename = 'elbow_method_plot.png'
plt.savefig(elbow_filename)


kmeans = KMeans(n_clusters=2)
kmeans.fit(data)

# Scatter plot with cluster labels
plt.scatter(x, y, c=kmeans.labels_)
plt.title('Scatter Plot with Cluster Labels')
plt.xlabel('X')
plt.ylabel('Y')
clustered_filename = 'clustered_scatter_plot.png'
plt.savefig(clustered_filename)
