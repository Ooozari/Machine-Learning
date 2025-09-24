import pandas as pd

df = pd.read_csv("./dataSet/Admission_Predict.csv")

columns=df.columns
df.drop("Serial No.", axis=1, inplace=True)
y=df["Chance of Admit "]
df.drop("Chance of Admit ", axis=1, inplace=True)
print(df.head())

