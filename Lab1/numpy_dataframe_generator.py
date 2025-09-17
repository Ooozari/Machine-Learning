import pandas as pd
import numpy as np

np.random.seed(42)
data =  {
    'Name': range(1,101),
    'Age': np.random.randint(50,100,100),
    'City': np.random.normal(70,10,100).round(1),
    'Salary':np.random.normal(70,5,100).round(1),
}

df_numpy = pd.DataFrame(data)
print('Numpy generated Data:')
print(df_numpy.head())