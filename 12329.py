# -*- coding: utf-8 -*-


import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os
os.environ["OMP_NUM_THREADS"] = '1'

from sklearn.cluster import KMeans



"""### Loading Data"""

df=pd.read_csv("Mall_Customers.csv")

"""The data includes the following features:

1. Customer ID
2. Customer Gender
3. Customer Age
4. Annual Income of the customer (in Thousand Dollars)
5. Spending score of the customer (based on customer behaviour and spending nature)
"""

df

df.head()

"""### Data Exploration"""

### Check Null Values

df.isnull().sum()

### Observation: There is no missing values.

### Visual and Statistical Understanding of data

df.columns

plt.scatter(df['Age'],df['Spending Score (1-100)'])
plt.xlabel("Age")
plt.ylabel("Spending Score")
plt.show()

### Observation: It seems to purpose two types of Customer

plt.scatter(df["Age"],df["Annual Income (k$)"])
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")

### Observation: No Group

plt.scatter(df["Spending Score (1-100)"], df["Annual Income (k$)"])
plt.xlabel("Spending Score (1-100)")
plt.ylabel("Annual Income (k$)")

### It seems to purpose five Groups

"""### Choose Relevant Columns

All the columns are  not relevant for the clustering. In this example, we will use the numerical ones: Age, Annual Income, and Spending Score
"""

relevant_cols = ["Age", "Annual Income (k$)",
                 "Spending Score (1-100)"]

customer_df = df[relevant_cols]

customer_df

"""### Data Transformation

Kmeans is sensitive to the measurement units and scales of the data. It is better to standardize the data first to tackle this issue

The standardization substracts the mean of any feature from the actual values of that feature and divides the feature’s standard deviation.
"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(customer_df)

scaled_data = scaler.transform(customer_df)

scaled_data

"""### Determine the best number of cluster"""

def find_best_clusters(df, maximum_K):
    clusters_centers = []
    k_values = []
    for k in range(2, maximum_K):
        kmeans_model = KMeans(n_clusters = k)
        kmeans_model.fit(df)

        clusters_centers.append(kmeans_model.inertia_)
        k_values.append(k)

    return clusters_centers, k_values

clusters_centers, k_values = find_best_clusters(scaled_data, 12)

def generate_elbow_plot(clusters_centers, k_values):

    figure = plt.subplots(figsize = (12, 6))
    plt.plot(k_values, clusters_centers, 'o-', color = 'orange')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Cluster Inertia")
    plt.title("Elbow Plot of KMeans")
    plt.show()

generate_elbow_plot(clusters_centers, k_values)

"""From the plot, we notice that the cluster inertia decreases as we increase the number of clusters. Also the drop the inertia is minimal after K=5 hence 5 can be considered as the optimal number of clusters.

### Create the final KMeans model
"""

kmeans_model = KMeans(n_clusters = 5)

kmeans_model.fit(scaled_data)

### We can access the cluster to which each data point belongs by using the .labels_ attribute.

df["clusters"] = kmeans_model.labels_

df

"""### Visualize the clusters"""

plt.scatter(df["Spending Score (1-100)"],
            df["Annual Income (k$)"],
            c = df["clusters"]
            )
