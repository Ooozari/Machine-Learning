# -------------Lab 5 - KMeans & DBSCAN Clustering-------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Step 1: Load dataset
data = pd.read_csv("../Dataset/Mall_Customers.csv")
print("Dataset Preview:\n", data.head())
print("\nDataset Info:\n", data.info())

# Select features
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Step 2: Apply K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
data['KMeans_Cluster'] = kmeans.fit_predict(X)

# Plot KMeans Clusters
plt.figure(figsize=(8,6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', 
                hue='KMeans_Cluster', data=data, palette='tab10', s=60)
plt.title('K-Means Clustering (Before Scaling)')
plt.show()

# Step 3: Apply DBSCAN
dbscan = DBSCAN(eps=5, min_samples=5)
data['DBSCAN_Cluster'] = dbscan.fit_predict(X)

plt.figure(figsize=(8,6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', 
                hue='DBSCAN_Cluster', data=data, palette='tab10', s=60)
plt.title('DBSCAN Clustering (Before Scaling)')
plt.show()

# Step 4: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Re-run KMeans on scaled data
kmeans_scaled = KMeans(n_clusters=5, random_state=42)
data['KMeans_Scaled_Cluster'] = kmeans_scaled.fit_predict(X_scaled)

# Plot scaled KMeans results
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_scaled[:,0], y=X_scaled[:,1], 
                hue=data['KMeans_Scaled_Cluster'], palette='tab10', s=60)
plt.title('K-Means Clustering (After Scaling)')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.show()

# Step 5: Compare Results
print("\nCluster comparison before and after scaling:")
print(data[['KMeans_Cluster', 'KMeans_Scaled_Cluster']].head())

# Step 6: Find the best 'k' using the Elbow Method
wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method to find optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Step 7: Interpret clusters
print("\nCluster Centers (Scaled Data):")
print(kmeans_scaled.cluster_centers_)
print("\nSilhouette Score (KMeans Scaled):", silhouette_score(X_scaled, data['KMeans_Scaled_Cluster']))

# Step 8: Add fake feature and test effect
np.random.seed(42)
data['LuckyNumber'] = np.random.randint(0, 101, size=len(data))

X_with_luck = data[['Annual Income (k$)', 'Spending Score (1-100)', 'LuckyNumber']]
kmeans_lucky = KMeans(n_clusters=5, random_state=42)
data['Cluster_with_Lucky'] = kmeans_lucky.fit_predict(X_with_luck)

plt.figure(figsize=(8,6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='Cluster_with_Lucky', data=data, palette='tab10', s=60)
plt.title('K-Means Clustering with LuckyNumber Feature')
plt.show()

print("\nAdding 'LuckyNumber' introduced random noise to clustering and changed boundaries — showing that irrelevant features can distort cluster accuracy.")

