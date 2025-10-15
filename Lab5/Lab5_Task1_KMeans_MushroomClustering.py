# ----------------------------K-Means Clustering - Unsupervised Learning----------------------------

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
import seaborn as sns

# Step 1: Load the dataset
mushroom_data = pd.read_csv("../Dataset/mushrooms.csv")

# Display first few rows
print("Dataset Preview :")
print(mushroom_data.head())

# Step 2: Check for the missing values
print("\n Missing Values in Each Columns :")
print(mushroom_data.isnull().sum())
print(f"Dataset shape: {mushroom_data.shape}")
print(f"Target distibution: \n {mushroom_data['class'].value_counts()}")


# ========EXPLORATORY DATA ANALYSIS========
plt.figure(figsize=(12,8))
plt.subplot(2,3,1)
sns.countplot(data=mushroom_data, x = 'class')
plt.title('Target distribution ')

plt.subplot(2,3,2)
sns.countplot(data=mushroom_data, x = 'cap-shape', hue='class')
plt.title("Cap Shape Vs. Class")
plt.subplot(2,3,3)
sns.countplot(data=mushroom_data, x = 'cap-color', hue='class')
plt.title("Cap Color Vs. Class")

plt.subplot(2,3,4)
sns.countplot(data=mushroom_data, x = 'population')
plt.title('Population Distribution')
plt.subplot(2,3,5)
sns.countplot(data=mushroom_data, x = 'habitat')
plt.title("Habitat Distribution")
plt.subplot(2,3,6)
sns.countplot(data=mushroom_data, x = 'odor', hue='class')
plt.title('Odor Vs. Class')

plt.tight_layout()
plt.show()

# ========== Data Preprocessing ==========

#handle the missing values
mushroom_df = mushroom_data.dropna()
 
#Separate features and target 

mushroom_features = mushroom_df.drop('class', axis =1)
mushroom_labels = mushroom_df['class']

#Encode the categorical features
encoder = OrdinalEncoder()
mushroom_prepared = encoder.fit_transform(mushroom_features)
mushroom_prep_df = pd.DataFrame(mushroom_prepared, columns=mushroom_features.columns)

# KMEANS CLustering 
# Create and fit the KMeans Model
k_clust = KMeans(n_clusters=2, random_state=42)

k_clust.fit(mushroom_prep_df)

# Get results
k_labels = k_clust.labels_

# Model Evaluations

# Map true labels to numerical values
map_dict = {'e' : 0, 'p':1 }
true_labels = mushroom_labels.map(map_dict)

#calculate accuracy 

accuracy = accuracy_score(true_labels, k_labels)
print(f"KMean Accuracy: {accuracy:.4f}")

# Confusion matrix

plt.figure(figsize=(8,6))
cm = confusion_matrix(true_labels, k_labels)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Cluster 0','Cluster 1'], 
yticklabels=['Edible','Poisonous'])

plt.title("Confusion Matrix")
plt.xlabel("Predicted Clusters")
plt.ylabel("True Labels")
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(true_labels, k_labels,
target_names = ['Edible','Poisonous']))












