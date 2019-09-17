import pandas as pd
import numpy as np
import sklearn
import sklearn.metrics
from sklearn.metrics import roc_auc_score as ras
from sklearn.metrics import precision_recall_curve as prc

df=pd.read_csv('scores.csv')
df0=df.drop(df.columns[[1,2,3,4]],axis=1)
df1=df.drop(df.columns[[0,2,3,4]],axis=1)
df2=df.drop(df.columns[[0,1,3,4]],axis=1)
df3=df.drop(df.columns[[0,1,2,4]],axis=1)
df4=df.drop(df.columns[[0,1,2,3]],axis=1)
#print(  ras(df0,df1),ras(df0,df2),ras(df0,df3),ras(df0,df4)   )

precision, recall, thresholds = prc(df0,df4)
#print(precision,'\n\n', recall,'\n\n', thresholds)
p=[]
for i in range (13):
    if recall[i] >= 0.7 :
        p.append(precision[i])
print(max(p))
