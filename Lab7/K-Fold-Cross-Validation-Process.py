from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    RepeatedKFold,
    RepeatedStratifiedKFold
)

# 1. Normal K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 2. Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 3. Repeated K-Fold
rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

# 4. Repeated Stratified K-Fold
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)


#  Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report

# Step 1: Load and prepare the data 
df = pd.read_csv("E:/University/7th Semester/ML/Lab/Dataset/titanic.csv")[['Survived', 'Pclass', 'Gender', 'Age', 'SibSp', 'Parch', 'Fare']].dropna()


# Convert 'Gender' to numeric (male=1, female=0)
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])


# Split features and target
X = df.drop('Survived', axis=1)
y = df['Survived']
#  Step 2: Define model and K-Fold 
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = RandomForestClassifier(random_state=42)

# Step 3: Cross-validation for accuracy 
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
print("Cross-Validation Accuracies:", scores)
print("Average Accuracy:", scores.mean())

#  Step 4: Generate predictions using cross-validation 
y_pred = cross_val_predict(model, X, y, cv=kf)
y_prob = cross_val_predict(model, X, y, cv=kf, method='predict_proba')[:, 1]


#  Step 5: Evaluate model performance 
print("\nConfusion Matrix:\n", confusion_matrix(y, y_pred))
print("\nClassification Report:\n", classification_report(y, y_pred))
print("ROC AUC Score:", roc_auc_score(y, y_prob))
#  Step 6: Plot the ROC Curve 
fpr, tpr, _ = roc_curve(y, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y, y_prob):.2f}")
plt.plot([0, 1], [0, 1], 'r--') # Random guess line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()