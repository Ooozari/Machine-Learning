# Lab 4 - Task 4: KNN for Credit Card Fraud Detection
# Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# 1. Download dataset from KaggleHub
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
print("Path to dataset files:", path)

# 2. Load dataset
df = pd.read_csv(f"{path}/creditcard.csv")
print("Dataset shape:", df.shape)
print(df.head())

# 3. Split into features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# 4. Handle imbalance using SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)
print("\nAfter SMOTE:")
print("Non-Fraud samples:", sum(y_resampled == 0))
print("Fraud samples:", sum(y_resampled == 1))

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# 6. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Train initial KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# 8. Evaluate initial model
print("\nInitial Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")

# 9. Hyperparameter tuning using GridSearchCV
param_grid = {"n_neighbors": [3, 5, 7, 9, 11]}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring="f1", n_jobs=-1)
grid.fit(X_train, y_train)

best_knn = grid.best_estimator_
print("\nBest Parameters from GridSearchCV:", grid.best_params_)

# 10. Evaluate tuned model
y_pred_best = best_knn.predict(X_test)
print("\nTuned Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_best):.4f}")

# Optional: detailed performance report
print("\nClassification Report:\n", classification_report(y_test, y_pred_best))
