# подключаем гугл-диск
from google.colab import drive
drive.mount('/content/gdrive')

# загружаем необходимые библиотеки
import numpy as np
import pandas as pd

# загрузим исходные таблицы
df_products = pd.read_csv('/content/gdrive/MyDrive/products.csv')
df_transactions = pd.read_csv('/content/gdrive/MyDrive/transactions.csv')

# группируем, делая рейтинг "голосованием покупками"
group = df_transactions.groupby(['user_id', 'product_id']).size()

# прогоняем группировку через csv, что бы получить обычный датафрейм
group.to_csv("group.csv", header=False)
df = pd.read_csv("group.csv", names=['user_id', 'product_id', 'rating'], dtype=int)

# смотрим количество уникальных клиентов
len(df.user_id.unique())

from surprise import Dataset
from surprise import Reader

# далее с помощью библиотеки Surprise сделаем предсказание

reader = Reader(rating_scale=(1, 99)) # Зададим разброс оценок
data = Dataset.load_from_df(df, reader) #создадим объект, с которым умеет работать библиотека

trainset = data.build_full_trainset()
testset = trainset.build_testset()

from surprise import SVD

algo = SVD(n_factors=7,random_state=42)
predictions = algo.fit(trainset).test(testset)

# переведем предсказание в формат pandas
df = pd.DataFrame(predictions)
df = df[['uid', 'iid', 'est']]
df.columns = ['user_id', 'product_id', 'rating']

df = df_ratings

from datetime import datetime
import time

# датафрейм для формирования csv на kaggle
df_answer = pd.DataFrame(columns=['user_id', 'product_id'])
# включаем замер времени
start_time = datetime.now()
for i in df['user_id'].unique():
    # пишем 10 рекомендованных продуктов для каждого клиента
    df_answer = df_answer.append({'user_id':i, 'product_id':str(df[['user_id', 'product_id']][df['user_id']==i].groupby(['user_id', 'product_id']).size().sort_values(ascending = False).head(10)[i].index.values)[1:-1].lstrip()}, ignore_index=True)
    break
print(datetime.now() - start_time)

# выгрузим полученный результат
df_answer.to_csv('/content/gdrive/MyDrive/answer.csv', index=False)
