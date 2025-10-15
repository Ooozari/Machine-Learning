import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = pd.read_csv("../Dataset/mushrooms.csv")

# display first few rows
print("Dataset preview :")
print(data.head())

# Step 2: Check for missing values
print("\nMissing Values in Each Column :")
print(data.isnull().sum())

# step 3 : Encode Categorical features (if any)
label_encoders = {}
for column in data.columns:
    if data[column].dtype == 'object':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

# step 4: Aplly Kmean clustering
Kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = Kmeans.fit_predict(data)

#Step 5: Display Cluster center and counts 
print("\nCluster Center:\n",Kmeans.cluster_centers_)
print("\nCluster Counts:\n", data['Cluster'].value_counts())

# Step 6: Visualize clusters using PCA (2D)
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data.drop('Cluster', axis=1))
plt.figure(figsize=(8,6))
plt.scatter(reduced_data[:,0], reduced_data[:,1], c=data['Cluster'], cmap='viridis', s=50)
plt.title('K-Means Clustering Visualization')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

