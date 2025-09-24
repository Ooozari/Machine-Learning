# Data Cleaning & Outlier Detection with Visualization
#
# Dataset Columns:
#   - Name
#   - Age
#   - Salary
#   - Department
#
# Tasks:
# 1. Handle missing values properly.
# 2. Encode the categorical feature Department.
# 3. Detect outliers in the Salary column using boxplot visualization.
# 4. Comment on whether Zara’s salary should be treated as an outlier or not.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Step 1: Create dataset
data = {
    "Name": ["Ahsan", "Hira", "Bilal", "Zara", "Salman", "Mahnoor"],
    "Age": [25, 27, 35, 29, None, 40],
    "Salary": [50000, None, 75000, 2000000, 60000, 90000],
    "Department": ["IT", "Finance", "IT", "HR", "Finance", "IT"]
}

df = pd.DataFrame(data)
print("Original Dataset:\n", df)

# Step 2: Handle missing values
# Fill missing Age with mean
df["Age"] = df["Age"].fillna(df["Age"].mean())

# Fill missing Salary with median
df["Salary"] = df["Salary"].fillna(df["Salary"].median())

print("\nAfter Handling Missing Values:\n", df)

# Step 3: Encode categorical Department feature
label_encoder = LabelEncoder()
df["Department_LabelEncoded"] = label_encoder.fit_transform(df["Department"])

print("\nAfter Encoding Department:\n", df)

# Step 4: Detect outliers in Salary using boxplot
plt.figure(figsize=(8,6))
plt.boxplot(df["Salary"], vert=False, patch_artist=True, boxprops=dict(facecolor="lightblue"))
plt.title("Salary Distribution with Outlier Detection")
plt.xlabel("Salary")
plt.show()

# Step 5: Outlier detection logic
q1 = df["Salary"].quantile(0.25)
q3 = df["Salary"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = df[(df["Salary"] < lower_bound) | (df["Salary"] > upper_bound)]
print("\nDetected Outliers in Salary:\n", outliers[["Name", "Salary"]])

# Step 6: Specific check for Zara
if "Zara" in outliers["Name"].values:
    print("\nComment: Zara’s salary is much higher than others and falls outside the IQR range,")
    print("so it should be treated as an outlier.")
else:
    print("\nComment: Zara’s salary is within normal range and should not be considered an outlier.")
