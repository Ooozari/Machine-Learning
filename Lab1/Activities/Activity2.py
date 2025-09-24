# Student Performance Tracker
#
# Dataset Columns:
#   - ID
#   - Math
#   - Science
#   - English
#   - Grade (A/B/C)
#
# Tasks:
# 1. Handle missing marks (fill with mean).
# 2. Encode the categorical Grade column.
# 3. Plot a histogram of Math scores with 10 bins.
# 4. Identify if any student has an unusually high/low total score using a boxplot.
# 5. Summarize findings (e.g., which grade performed best overall).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Step 1: Generate random dataset
np.random.seed(42)  # for reproducibility

students = pd.DataFrame({
    "ID": range(1, 101),
    "Math": np.random.randint(40, 100, 100).astype(float),   # random marks 40–100
    "Science": np.random.randint(35, 95, 100).astype(float),
    "English": np.random.randint(30, 90, 100).astype(float),
    "Grade": np.random.choice(["A", "B", "C"], size=100)    # random grades
})

# Introduce some missing values for demonstration
for col in ["Math", "Science", "English"]:
    students.loc[np.random.choice(students.index, 5, replace=False), col] = np.nan

print("Original Dataset (with some missing values):\n", students.head())

# Step 2: Handle missing marks (fill with mean of each subject)
students["Math"].fillna(students["Math"].mean(), inplace=True)
students["Science"].fillna(students["Science"].mean(), inplace=True)
students["English"].fillna(students["English"].mean(), inplace=True)

print("\nAfter Handling Missing Values:\n", students.head())

# Step 3: Encode Grade column
label_encoder = LabelEncoder()
students["Grade_Encoded"] = label_encoder.fit_transform(students["Grade"])
print("\nAfter Encoding Grade:\n", students[["Grade", "Grade_Encoded"]].head())

# Step 4: Plot histogram of Math scores
plt.figure(figsize=(8,6))
plt.hist(students["Math"], bins=10, color="skyblue", edgecolor="black")
plt.title("Histogram of Math Scores")
plt.xlabel("Math Score")
plt.ylabel("Number of Students")
plt.grid(axis="y")
plt.show()

# Step 5: Detect unusual total scores using boxplot
students["Total"] = students["Math"] + students["Science"] + students["English"]

plt.figure(figsize=(8,6))
sns.boxplot(x=students["Total"], color="lightgreen")
plt.title("Boxplot of Total Scores")
plt.xlabel("Total Score")
plt.show()

# Outlier detection using IQR
q1 = students["Total"].quantile(0.25)
q3 = students["Total"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = students[(students["Total"] < lower_bound) | (students["Total"] > upper_bound)]
print("\nUnusually High/Low Total Scores:\n", outliers[["ID", "Total", "Grade"]])

# Step 6: Summarize findings (average total score per grade)
summary = students.groupby("Grade")["Total"].mean().sort_values(ascending=False)
print("\nAverage Total Score by Grade:\n", summary)
best_grade = summary.idxmax()
print(f"\n✅ Overall, Grade {best_grade} performed the best on average.")
