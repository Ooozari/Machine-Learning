import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25,23,5,3],
    'City': ['Lhr', 'Fsd', 'Isb', 'Khi'],
    'Salary': [1000,5000,2000,3000]
}

myData = pd.DataFrame(data)
print("My manual data frame: ")
print(myData)