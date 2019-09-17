import  numpy   as  np
import  pandas  as  pd
import  sklearn
from sklearn import datasets
from sklearn import model_selection
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection    import KFold
from sklearn.neighbors          import KNeighborsClassifier
from sklearn.model_selection    import cross_val_score
from sklearn                    import preprocessing

Boston = sklearn.datasets.load_boston()
Bdata = pd.DataFrame(Boston.data)
X =sklearn.preprocessing.scale(np.array(Bdata))
y= np.array(pd.DataFrame(Boston.target))

kf=model_selection.KFold(n_splits=5,random_state=42,shuffle=True)
kf.get_n_splits(X)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

z= np.linspace(1,10,num=200,endpoint=True)

for i in range(0,200):
   
    neigh = KNeighborsRegressor(n_neighbors=5,weights='distance',p=z[i])
    for train_index in kf.split(X):    
        neigh.fit(X_train, np.ravel(y_train,order='C'))        
    scores = cross_val_score(neigh ,X_train, np.ravel(y_train), cv=kf, scoring='neg_mean_squared_error')    
    print(scores.mean(),z[i])


