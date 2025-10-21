import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_wine
import matplotlib as plt
# Load the wine data set
wine = load_wine()
 
#Create the dataset
wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)

# Add target class
wine_df['target'] = wine.target


# Explore the dataset

print("Dataset info : ",wine_df.info())
print("\nShape of dataset: ",wine_df.shape)
print("Number of features: ", len(wine.feature_names))

print("Total Obervations: ", len(wine_df))


print("Data types :",wine_df.dtypes)

# checking for missing values 
print("\n Missing Values: ",wine_df.isnull().sum())

# filling the values
wine_df.fillna(0, inplace=True)


print("\nSummary Statistics:\n", wine_df.describe())


# Boxplot for visual inspection 

plt.figure(figsize=(10,8))
wine_df.boxplot(rot=90)
plt.title("Boxplots for Outlier Detection")
plt.show()

