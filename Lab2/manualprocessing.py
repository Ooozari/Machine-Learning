## Calculating Manually the y = mx + b
import pandas as pd
df = pd.read_csv("./dataSet/headbrain.csv")
print(df.head())

# Computing X and Y
X = df['Head Size(cm^3)'].values
y = df['Brain Weight(grams)'].values

import numpy as np
X_sum = np.sum(X)
X_squared_sum = np.sum(X * X)
n = len(X)
print("Sum of X: ", X_sum)
print("Sum of X Squared: ", X_squared_sum)
print("Length of X: ", n)


#calculating cross-derivation and derviation about x
numer = n*np.sum(X*y)-np.sum(X)*np.sum(y)
denom = n*np.sum(X*X)-(np.sum(X))**2
w1 = numer/denom

#calculating regression coefficients
w0 = (np.sum(y)-w1*(np.sum(X)))/n
print("W1: ", w1)
print("W0: ", w0)

import matplotlib.pyplot as plt

max_x = np.max(X)
min_x = np.min(X)

#Calculating lines values x and y
x1 = np.linspace(min_x,max_x)
y1 = w0 + w1 * x1
# Ploting Line
plt.plot(x1, y1, color='red', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X,y,c='green', label='Scatter Plot')
plt.xlabel('Head Size in cm^3')
plt.ylabel('Brain weight in grams')
plt.legend()
plt.show()

# Calculating RMSE
rmse = 0
for i in range(n):
    y_pred = w0 + w1 * X[i]
    rmse += (y[i] - y_pred) ** 2
rmse = np.sqrt(rmse/n)
print("RMSE =", rmse)

# Calculating R^2

ss_tot = 0
ss_res = 0

y_mean = np.mean(y)

for i in range(n):
    y_pred = w0 + w1 * X[i]
    ss_tot += (y[i]- y_mean) ** 2
    ss_res += (y[i]- y_pred) ** 2
r2 = 1 - (ss_res/ss_tot)
print("R2 Score =",r2)

