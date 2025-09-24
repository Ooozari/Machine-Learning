import pandas as pd

df = pd.read_csv("./dataSet/Admission_Predict.csv")

columns=df.columns
df.drop("Serial No.", axis=1, inplace=True)
y=df["Chance of Admit "]
df.drop("Chance of Admit ", axis=1, inplace=True)
print(df.head())

# Implementing the train_test_split
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(df,y,test_size=0.2)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train,y_train)
pred = lr.predict(x_test)
print(pred)
