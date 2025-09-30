import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
companies = pd.read_csv("./dataSet/1000_Companies.csv")
columns = companies.columns
print(companies.head())

# Independent variables (all except last column "Profit")
X = companies.iloc[:, :-1].values  

# Dependent variable (last column "Profit")
y = companies.iloc[:, -1].values  

# Correlation heatmap
sns.heatmap(companies.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.show()

# Encoding categorical data (State column at index 3)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])

# Splitting the dataset into Training and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
print("Predicted:", y_pred[:5])
print("Actual:", y_test[:5])

# Calculating the R square value and the mean squared error
from sklearn.metrics import r2_score, mean_squared_error
print("R square value: %.2f" % r2_score(y_test, y_pred))
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
