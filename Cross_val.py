import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection    import KFold
from sklearn.metrics            import accuracy_score
from sklearn.neighbors          import KNeighborsClassifier
from sklearn.model_selection    import cross_val_score
from sklearn.model_selection    import train_test_split
from sklearn                    import preprocessing

data = pd.read_csv('wine.data')
dataCl = pd.DataFrame(data['1'])
data = data.drop(data.columns[0],axis='columns')
X=preprocessing.scale(np.array(data))
y=np.array(dataCl)
kf=KFold(n_splits=5,random_state=42,shuffle=True)
kf.get_n_splits(X)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

z=[]
for k in range(1,51):
    neigh = KNeighborsClassifier(n_neighbors=k)
    for train_index in kf.split(X):    
        neigh.fit(X_train, np.ravel(y_train,order='C'))
        
    scores = cross_val_score(neigh ,X_train, np.ravel(y_train), cv=kf, scoring='accuracy')
    print(scores.mean(),k)
    z.append(scores.mean())
print(sorted(z))
