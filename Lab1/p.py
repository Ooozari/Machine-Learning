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



#################
import pandas as pd
from sklearn.preprocessing import LabelEncoder

feature_df  = pd.DataFrame({'Regionname' : ['North', 'South', 'East','West', 'North']})

le = LabelEncoder()

feature_df['Region'] = le.fit_transform(feature_df['Regionname'])

print("Label Encoded Regions ",feature_df.value_counts())