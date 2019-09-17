import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits

data= pd.read_csv('titanic(2).csv', index_col='PassengerId')
data=data.drop(data.columns[[2,5,6,7,9,10]], axis='columns')
print(data.columns)
data=data.dropna()

Survived=pd.DataFrame(data['Survived'])
Survived=Survived.values

dataFemale=pd.DataFrame(data['Sex'].replace(regex='female', value=0))
dataMale=pd.DataFrame(dataFemale['Sex'].replace(regex='male', value=1))
data.update(dataMale)
data=data.drop(data.columns[0], axis='columns')
print(data)

Vse=pd.DataFrame(data)
Vse=Vse.values
print(Vse)


clf = DecisionTreeClassifier(random_state=241)
X = np.array(Vse)
y = np.array(Survived)
clf.fit(X, y)
importances = clf.feature_importances_
print(importances)
