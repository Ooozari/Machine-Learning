# Pakistani Provinces Analysis
#
# Dataset Columns:
#   - Province
#   - Population (millions)
#   - Literacy Rate (%)
#   - Region (North, South, East, West)
#
# Tasks:
# 1. Handle any missing values (drop one row, fill another with median).
# 2. Encode the Region column (Label Encoding + One-Hot Encoding).
# 3. Visualize population vs literacy rate using a scatter plot (label each province).
# 4. Detect if there is any province behaving like an outlier in literacy rate.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Step 1: Create dataset
data = {
    "Province": ["Punjab", "Sindh", "Khyber Pakhtunkhwa", "Balochistan"],
    "Population (millions)": [110, 47, 35, 12],
    "Literacy Rate (%)": [64, 58, np.nan, 46],  # Added missing value
    "Region": ["East", "South", "North", np.nan]  # Added missing region
}

df = pd.DataFrame(data)
print("Original Dataset:\n", df)

# Step 2: Handle missing values
# Drop row with missing Region (Balochistan)
df_drop = df.dropna(subset=["Region"])

# Fill missing Literacy Rate with median (using .loc to avoid warnings)
median_lit = df_drop["Literacy Rate (%)"].median()
df_drop.loc[:, "Literacy Rate (%)"] = df_drop["Literacy Rate (%)"].fillna(median_lit)

print("\nAfter Handling Missing Values:\n", df_drop)

# Step 3: Encode Region column
# Label Encoding
label_encoder = LabelEncoder()
df_drop.loc[:, "Region_LabelEncoded"] = label_encoder.fit_transform(df_drop["Region"])

# One-Hot Encoding (compatible with old/new sklearn)
try:
    onehot_encoder = OneHotEncoder(sparse_output=False)  # sklearn >= 1.2
except TypeError:
    onehot_encoder = OneHotEncoder(sparse=False)  # sklearn < 1.2

onehot = onehot_encoder.fit_transform(df_drop[["Region"]])
onehot_df = pd.DataFrame(onehot, columns=onehot_encoder.get_feature_names_out(["Region"]))
df_encoded = pd.concat([df_drop.reset_index(drop=True), onehot_df], axis=1)

print("\nAfter Encoding:\n", df_encoded)

# Step 4: Visualization
plt.figure(figsize=(8,6))
plt.scatter(df_encoded["Population (millions)"], df_encoded["Literacy Rate (%)"], color="orange", s=100)

# Add labels to each point
for i, txt in enumerate(df_encoded["Province"]):
    plt.annotate(txt, 
                 (df_encoded["Population (millions)"].iloc[i], 
                  df_encoded["Literacy Rate (%)"].iloc[i]), 
                 textcoords="offset points", xytext=(5,5))

plt.xlabel("Population (millions)")
plt.ylabel("Literacy Rate (%)")
plt.title("Population vs Literacy Rate of Pakistani Provinces")
plt.grid(True)
plt.show()

# Step 5: Outlier Detection (simple check using IQR)
q1 = df_encoded["Literacy Rate (%)"].quantile(0.25)
q3 = df_encoded["Literacy Rate (%)"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = df_encoded[(df_encoded["Literacy Rate (%)"] < lower_bound) | 
                      (df_encoded["Literacy Rate (%)"] > upper_bound)]

print("\nPossible Outliers in Literacy Rate:\n", outliers[["Province", "Literacy Rate (%)"]])
