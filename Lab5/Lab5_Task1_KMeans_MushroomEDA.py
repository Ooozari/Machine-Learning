# ----------------------------K-Means Clustering - Unsupervised Learning----------------------------

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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










