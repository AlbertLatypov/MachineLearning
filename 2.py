import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA

d_j = pd.read_csv('djia_index.csv')
d_j = d_j.drop(columns=['date'])
c_p = pd.read_csv('close_prices.csv')
c_p = c_p.drop(columns=['date'])

pca = PCA(n_components=1)
pca.fit(c_p)
print(pca.explained_variance_ratio_)

component1 = pca.transform(c_p)
m = pd.DataFrame(component1)
m = m.merge(d_j,left_index=True,right_index=True)
print (m.corr(method='pearson'))

print(pca.components_)



