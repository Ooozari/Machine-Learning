import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

data = {
    'A': [1,2,None,4,5],
    'B': [6,None,8,9,10],
    'C': [11,12,13,None,15],
}

df= pd.DataFrame(data)
print("Original Data Frame: ")

print(df)


