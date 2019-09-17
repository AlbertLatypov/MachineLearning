import numpy as np
import pandas as pd
import sklearn
data= pd.read_csv('svm-data.csv',header=-1)
print(data)
X = data.drop(data.columns[0],axis='columns')
y = data.drop(data.columns[[1,2]],axis='columns')

from sklearn.svm import SVC
clf = SVC(C = 100000,kernel='linear',random_state=241)
clf.fit(X, np.ravel(y))
print(clf.support_)
