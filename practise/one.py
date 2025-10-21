import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("../Dataset/1000_Companies.csv")
print("Printing the dataset :", data.head())

cols = data.columns.tolist()
print("The names of columns : ",cols)
print(f"Number of rows and columns : {data.shape}")
print("Data types : ", data.dtypes)

print("\nMissing Values in each column:")
print(data.isnull().sum())

# Identifying the Categorical Columns 
categorical_cols = data.select_dtypes(include=['object']).columns
print("\nCategorical Columns : ", categorical_cols)


le = LabelEncoder()
data["State"] = le.fit_transform(data['State'])
print("\nAfter Encoding 'State' Column:")
print(data.head())

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Apply scaling only to numeric columns (excluding Profit since it's target)
features = data.drop('Profit', axis=1)
scaled_features = scaler.fit_transform(features)

scaled_df = pd.DataFrame(scaled_features, columns=features.columns)

print("\nScaled Features (First 5 rows):")
print(scaled_df.head())


final_df = pd.concat([scaled_df, data['Profit']], axis=1)
print("\nFinal Processed Dataset:")
print(final_df.head())
