# Multi-format Data Loading & Visualization Challenge
#
# Tasks:
# 1. Load CSV, JSON, and Excel files.
# 2. Merge into single DataFrame (Name, Marks, Attendance, Bonus).
# 3. Scatter plot of Marks vs Attendance (highlight <70% attendance).
# 4. Apply one-hot encoding on Name.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the datasets
students_df = pd.read_csv("./dataset/students.csv")
attendance_df = pd.read_json("./dataset/attendance.json")
bonus_df = pd.read_excel("./dataset/extra.xlsx")


print("Students Data:\n", students_df.head())
print("\nAttendance Data:\n", attendance_df.head())
print("\nBonus Data:\n", bonus_df.head())

# Step 2: Merge into a single DataFrame
df = students_df.merge(attendance_df, on="Name").merge(bonus_df, on="Name")
print("\nMerged DataFrame:\n", df.head())

# Step 3: Scatter plot of Marks vs Attendance
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x="Marks", y="Attendance", hue=df["Attendance"] < 70, palette={True: "red", False: "green"}, s=100)
plt.title("Marks vs Attendance (Red = <70%)")
plt.xlabel("Marks")
plt.ylabel("Attendance (%)")
plt.legend(title="Low Attendance (<70%)")
plt.grid(True)
plt.show()

# Step 4: One-hot encode the Name column
encoded_df = pd.get_dummies(df, columns=["Name"])
print("\nDataFrame with One-Hot Encoded Names:\n", encoded_df.head())
