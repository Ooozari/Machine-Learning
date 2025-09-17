import pandas as pd

df_csv = pd.read_csv('./dataset/cars.csv')
print('Cars Csv Data:')
print(df_csv.head())

online_json_url = 'https://jsonplaceholder.typicode.com/users'
df_online_json = pd.read_json(online_json_url)
print('Online JSON Data:')
print(df_online_json.head())
