import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

data = pd.read_csv('perceptron-train.csv',header=[-1])
test = pd.read_csv('perceptron-test.csv',header=[-1])

X = data.drop(data.columns[0],axis='columns')
y = data.drop(data.columns[[1,2]],axis='columns')

Xtest=np.array(test.drop(test.columns[0],axis='columns'))
ytest=np.array(test.drop(test.columns[[1,2]],axis='columns'))


plt.scatter(X[2], y[0]);
plt.xlabel('Класс')
plt.ylabel('Признак');
#plt.show()


X = np.array(X)
y = np.array(y)
clf = Perceptron(max_iter=5,tol=1e-3,random_state=241)
clf.fit(X, np.ravel(y))              #обучение
predictions = clf.predict(Xtest)    #тест
print(accuracy_score(ytest,predictions))

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(Xtest)
clf.fit(X_train_scaled, np.ravel(y))
predictions = clf.predict(X_test_scaled)
print(accuracy_score(ytest,predictions))
