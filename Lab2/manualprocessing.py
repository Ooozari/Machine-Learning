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


