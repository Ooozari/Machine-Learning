from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    RepeatedKFold,
    RepeatedStratifiedKFold
)


kf = KFold(n_splits=5, shuffle=True, random_state=42)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)


rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report

df = pd.read_csv("E:/University/7th Semester/ML/Lab/Dataset/titanic.csv")[['Survived', 'Pclass', 'Gender', 'Age', 'SibSp', 'Parch', 'Fare']].dropna()


df['Gender'] = LabelEncoder().fit_transform(df['Gender'])



X = df.drop('Survived', axis=1)
y = df['Survived']
 
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = RandomForestClassifier(random_state=42)


scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
print("Cross-Validation Accuracies:", scores)
print("Average Accuracy:", scores.mean())

 
y_pred = cross_val_predict(model, X, y, cv=kf)
y_prob = cross_val_predict(model, X, y, cv=kf, method='predict_proba')[:, 1]



print("\nConfusion Matrix:\n", confusion_matrix(y, y_pred))
print("\nClassification Report:\n", classification_report(y, y_pred))
print("ROC AUC Score:", roc_auc_score(y, y_prob))

fpr, tpr, _ = roc_curve(y, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y, y_prob):.2f}")
plt.plot([0, 1], [0, 1], 'r--') 
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()