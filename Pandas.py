import numpy as np
import pandas as pd
data = pd.read_csv('titanic(2).csv', index_col='PassengerId')
print (data.columns)
print (data.corr(method='pearson'))
print(round(0.4148,2))
#print(data.groupby(['Sex']).size())
#print (data.groupby(['Survived']).size())
#jopa= pd.DataFrame(data['Name'].replace(regex='^.+[,] ', value=''))
#print(jopa)
#jopa1= pd.DataFrame(jopa['Name'].replace(regex='^.*Mrs. ', value=''))
#print(jopa1)
#jopa2= pd.DataFrame(jopa1['Name'].replace(regex='^.*Miss. ', value=''))

