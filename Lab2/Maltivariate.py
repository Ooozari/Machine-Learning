import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

companies = pd.read_csv("./dataSet/1000_Companies.csv")
coulumns = companies.columns
print(companies.head())

X = companies.iloc[:,-1].values
y = companies.iloc[:,4].values
X[:5]
sns.heatmap(companies.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.show()

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder= LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])
X[:5]