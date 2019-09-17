import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold

newsgroups = datasets.fetch_20newsgroups(
                    subset='all', 
                    categories=['alt.atheism', 'sci.space']
             )

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(newsgroups.data)
y = newsgroups.target
feature_mapping = vectorizer.get_feature_names()

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X,y)
best = gs.best_estimator_
best.fit(X, y)
jopa = best.coef_
scipy.sparse.save_npz('jopa', jopa)

sm = scipy.sparse.load_npz('jopa.npz') 
row = np.absolute(  sm.getrow(0).toarray()[0].ravel()   )
top_ten_indicies = row.argsort()[-10:] # упорядочивает по элементам, а затем выводит индексы последних 10-и   
top_ten_values = row[row.argsort()[-10:]] # по индексам выводит сами элементы
for i in top_ten_indicies:
    print(feature_mapping[i])

