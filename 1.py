import pandas as pd
import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack

df=pd.read_csv('salary-train.csv')
df_test=pd.read_csv('salary-test-mini.csv')
target=df['SalaryNormalized']
df['FullDescription'] = df['FullDescription'].str.lower()
df['FullDescription'] = df['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
df_test['FullDescription'] = df_test['FullDescription'].str.lower()
df_test['FullDescription'] = df_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
df=df.drop(['SalaryNormalized'], axis=1)
df_test=df_test.drop(['SalaryNormalized'], axis=1)



vectorizer = TfidfVectorizer(min_df=5)
X = vectorizer.fit_transform(df['FullDescription'])
X_test=vectorizer.transform(df_test['FullDescription'])



enc = DictVectorizer()
for row in df.loc[df.ContractTime.isnull(), 'ContractTime'].index:
    df.at[row, 'ContractTime'] = 'sad23'
for row in df_test.loc[df_test.ContractTime.isnull(), 'ContractTime'].index:
    df_test.at[row, 'ContractTime'] = 'sad23'
X_train_categ = enc.fit_transform(df[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ=   enc.transform(df_test[['LocationNormalized', 'ContractTime']].to_dict('records'))
matr=hstack([X,X_train_categ])
matr1=hstack([X_test,X_test_categ])



clf = Ridge(random_state=241,alpha=1.0)
clf.fit(matr, target)
pred=clf.predict(matr1)
print(pred)
