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

#splitting the dataset into the Traning and Test set
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state= 0)

#fitting Multiple Linear Regression to the Traning ser
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)