import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#Load dataset
data = pd.read_csv("../Dataset/Position_Salaries.csv")

#independent variable (level) and dependent varibale (salary)
X = data[["Level"]].values
y =data["Salary"].values

#polynomial transformation [defree = 4 for better curve fitting]
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

#Train polynomail regression model
poly_model = LinearRegression()
poly_model.fit(X_poly,y)

#Predictions
y_pred = poly_model.predict(X_poly)

#Plot results
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, y_pred, color="red", label="Polynomial Fit (deg=4)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.title("Polynomial Regression")
plt.legend()
plt.show()

#calculating R2 score
r2 = r2_score(y,y_pred)
#display results
print("POLYNOMIAL REGRESSION METRIC:")
print(f"R² Score: {r2:.4f}")

