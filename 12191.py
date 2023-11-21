import random

def dist(point1, point2):
    # calculating distance between 2 points
    return sum((x-y) ** 2 for x, y in zip(point1, point2)) ** 0.5

def cluster_points(points, centroids):
    # assigning cluster points
    clusters = [[] for i in range(len(centroids))]
    
    for point in points:
        distances = [dist(point, centroid) for centroid in centroids]
        closest_cluster_index = distances.index(min(distances))
        clusters[closest_cluster_index].append(point)
    return clusters

def update_centroids(clusters):
    # to update centroids
    return [tuple(sum(x) / len(x) for x in zip(*cluster)) for cluster in clusters]

def k_means(points, k, max_iterations = 100):
    centroids = random.sample(points, k)  # randomly assiging 3 centroids
    
    for i in range(max_iterations):
        clusters = cluster_points(points, centroids)  # nearest centroid data point assingning
        
        # update centroids 
        new_centroids = update_centroids(clusters)
        
        if new_centroids == centroids:
            break
        
        centroids = new_centroids
    return centroids, clusters        

points = [(2,10), (2,5), (8,4), (5,8), (7,5), (6,4), (1,2), (4,9)]   # point co-ordinates
k = 3  # no of clusters

centroids, clusters = k_means(points, k)

for i, centroid in enumerate(centroids):
    print('Cluster', (i + 1), 'centroid :', centroid)
    print('Data points in Cluster', (i + 1), clusters[i])
    print()
