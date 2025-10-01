import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
data = pd.read_csv("../Dataset/Medical_Cost_Personal_Datasets.csv")  # adjust path if needed

# 2. Explore dataset
print("First 5 rows:\n", data.head())
print("\nDataset Info:\n")
print(data.info())
print("\nMissing values:\n", data.isnull().sum())

# 3. Preprocessing
# Encode categorical variables (region, smoker, sex)
data = pd.get_dummies(data, columns=["region", "smoker", "sex"], drop_first=True)

# Define features (X) and target (y)
X = data.drop("charges", axis=1).values
y = data["charges"].values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predictions
y_pred = model.predict(X_test)

# 6. Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# 7. Plot Predicted vs Actual Costs
plt.scatter(y_test, y_pred, alpha=0.7, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color="red", linewidth=2, label="Perfect Prediction")
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Predicted vs Actual Insurance Charges")
plt.legend()
plt.show()
