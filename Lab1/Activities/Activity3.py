# COVID-19 Dataset Exploration
#
# Tasks:
# 1. Load dataset from online source.
# 2. Extract only Pakistan’s data.
# 3. Handle missing values (if any).
# 4. Plot a line chart showing cases over time.
# 5. Find the day with the highest confirmed cases.
# 6. Detect outliers in daily cases using a boxplot.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load dataset
url = "https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv"
df = pd.read_csv(url)
print("Original Dataset Shape:", df.shape)

# Step 2: Extract Pakistan data
pakistan_df = df[df["Country/Region"] == "Pakistan"].copy()
print("\nPakistan Dataset:\n", pakistan_df.head())

# Step 3: Handle missing values (fill with 0 for cases)
pakistan_df.fillna(0, inplace=True)

# Step 4: Plot line chart of confirmed cases over time
plt.figure(figsize=(10,6))
plt.plot(pakistan_df["Date"], pakistan_df["Confirmed"], label="Confirmed Cases", color="red")
plt.xlabel("Date")
plt.ylabel("Confirmed Cases")
plt.title("COVID-19 Confirmed Cases in Pakistan")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()

# Step 5: Find the day with the highest confirmed cases
max_cases_row = pakistan_df.loc[pakistan_df["Confirmed"].idxmax()]
print("\n📌 Day with Highest Confirmed Cases:")
print(max_cases_row[["Date", "Confirmed"]])

# Step 6: Detect outliers in daily new cases using boxplot
pakistan_df["Daily_Cases"] = pakistan_df["Confirmed"].diff().fillna(0)

plt.figure(figsize=(8,6))
sns.boxplot(y=pakistan_df["Daily_Cases"], color="skyblue")
plt.title("Boxplot of Daily New Cases in Pakistan")
plt.ylabel("Daily Cases")
plt.show()

# Outlier detection using IQR
q1 = pakistan_df["Daily_Cases"].quantile(0.25)
q3 = pakistan_df["Daily_Cases"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = pakistan_df[(pakistan_df["Daily_Cases"] < lower_bound) | 
                       (pakistan_df["Daily_Cases"] > upper_bound)]

print("\n📌 Outlier Days (Unusual Daily Cases):\n", outliers[["Date", "Daily_Cases"]])
