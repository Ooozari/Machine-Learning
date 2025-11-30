import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
    classification_report
)


iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = RandomForestClassifier(random_state=42)


scores = cross_val_score(model, X, y, cv=kf, scoring="accuracy")
print("Cross-Validation Accuracies:", scores)
print("Average Accuracy:", scores.mean())


y_pred = cross_val_predict(model, X, y, cv=kf)
y_prob = cross_val_predict(model, X, y, cv=kf, method="predict_proba")


print("\nConfusion Matrix:\n", confusion_matrix(y, y_pred))


print("\nClassification Report:\n", classification_report(y, y_pred))


lb = LabelBinarizer()
y_bin = lb.fit_transform(y)

auc_score = roc_auc_score(y_bin, y_prob, multi_class="ovr")
print("ROC AUC Score (OvR):", auc_score)


for i in range(3):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
    plt.plot(fpr, tpr, label=f"Class {i} AUC = {roc_auc_score(y_bin[:, i], y_prob[:, i]):.2f}")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (Iris Dataset - OvR)")
plt.legend()
plt.show()
