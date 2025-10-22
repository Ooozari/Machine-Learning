# ==========================================
# LAB 6: PCA IMPLEMENTATION ON WINE DATASET
# ==========================================

# Import required libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


#1) Load and prepare dataset
wine = load_wine()
wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
wine_df['target'] = wine.target

print("Dataset Loaded Successfully!")
print("Shape of Dataset:", wine_df.shape)
print("\nMissing Values:\n", wine_df.isnull().sum())


#2) Data preprocessing
# Split features (X) and target (y)
X = wine_df.drop('target', axis=1)
y = wine_df['target']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nData Standardized — Mean:", round(X_scaled.mean(), 2),
      "| Std Dev:", round(X_scaled.std(), 2))


#3) Apply PCA
# Reduce data to 2 principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create new DataFrame with PCA results
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['target'] = y


#4) Explained variance ratio
print("\nExplained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Variance Captured:", sum(pca.explained_variance_ratio_))

# Bar chart for variance explained
plt.figure(figsize=(6,4))
sns.barplot(x=[1, 2], y=pca.explained_variance_ratio_, palette="coolwarm")
plt.title("Variance Explained by Each Principal Component")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.show()


#5) Visualize PCA in 2D
plt.figure(figsize=(10,8))
sns.scatterplot(
    data=pca_df,
    x='PC1',
    y='PC2',
    hue='target',
    palette='Set1',
    s=90,
    alpha=0.8
)
plt.title("PCA - 2D Projection of Wine Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Wine Class")
plt.show()


#6) Deploy PCA model on new data
sample = X.iloc[0:1]                  # Take one sample from dataset
sample_scaled = scaler.transform(sample)
sample_pca = pca.transform(sample_scaled)

print("\nPCA Transformed Sample:\n", sample_pca)
print("\nPCA Implementation Complete (Lab 6)")
