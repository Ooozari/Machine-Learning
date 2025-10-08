# Import all relevant libraries

from sklearn.svm import SVC
import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("../Dataset/income_evaluation.csv")

df.head()

print(df)

# identify categorical and numerical features
categorical_features = df.select_dtypes(include=['object']).columns
numerical_features = df.select_dtypes(exclude=['object']).columns

# Apply one-hot encoding to categorical features
df = pd.get_dummies(df, columns=categorical_features,drop_first=True)

# separate features and target variable again after one-hot encoding
X = df.drop("income_ >50K", axis=1)
y = df["income_ >50K"]

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=101)

#Scale numerical features 
scaler  = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])

X_test[numerical_features] = scaler.transform(X_test[numerical_features])
print(X_train.head())


svc = SVC(random_state=101)
svc.fit(X_train,y_train)

accuracies = cross_val_score(svc,X_train,y_train,cv=5)

print("Train Score:", np.mean(accuracies))

print("Test Score:", svc.score(X_test,y_test))
