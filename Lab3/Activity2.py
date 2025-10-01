import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
data = pd.read_csv("../Dataset/Car_Price.csv")

# 2. Explore dataset
print("First 5 rows:\n", data.head())
print("\nDataset Info:\n")
print(data.info())

# Select features and target (adapted to available dataset)
features = ["enginesize", "horsepower"]
target = "price"

X = data[features]
y = data[target]

# Handle missing values
X = X.dropna()
y = y.loc[X.index]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- LINEAR REGRESSION ----------
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
y_pred_lin = lin_reg.predict(X_test_scaled)

rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_lin))
r2_lin = r2_score(y_test, y_pred_lin)

print("\nLINEAR REGRESSION PERFORMANCE:")
print(f"RMSE: {rmse_lin:.2f}")
print(f"R² Score: {r2_lin:.4f}")

# ---------- POLYNOMIAL REGRESSION ----------
degrees = [2, 3, 4]
results = {}

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred_poly = model.predict(X_test_poly)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred_poly))
    r2 = r2_score(y_test, y_pred_poly)
    results[d] = (rmse, r2)

    print(f"\nPOLYNOMIAL REGRESSION (degree={d}):")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")

# ---------- PLOTS ----------
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_lin, color="blue", label="Linear Regression")
colors = ["red", "green", "purple"]
for i, d in enumerate(degrees):
    poly = PolynomialFeatures(degree=d)
    X_test_poly = poly.fit_transform(X_test_scaled)
    model = LinearRegression()
    model.fit(poly.fit_transform(X_train_scaled), y_train)
    y_pred_poly = model.predict(X_test_poly)
    plt.scatter(y_test, y_pred_poly, color=colors[i], alpha=0.5, label=f"Poly deg={d}")

plt.xlabel("Actual Car Prices")
plt.ylabel("Predicted Car Prices")
plt.title("Car Price Prediction: Linear vs Polynomial Regression")
plt.legend()
plt.show()
