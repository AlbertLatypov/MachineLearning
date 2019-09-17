import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score as ras

df=pd.read_csv('classification.csv')
matr=df.values
tp=0
for i in range (200):
    if matr[i][0]==1 and matr[i][1]==1 :
        tp+=1

fp=0
for i in range (200):
    if matr[i][0]==0 and matr[i][1]==1 :
        fp+=1

fn=0
for i in range (200):
    if matr[i][0]==1 and matr[i][1]==0 :
        fn+=1

tn=0
for i in range (200):
    if matr[i][0]==0 and matr[i][1]==0 :
        tn+=1

print(  (tp+tn)/(tp+fp+fn+tn)   ,\
        tp/(tp+fp)  ,\
        tp/(tp+fn)  ,\
        2*(tp/(tp+fp))*(tp/(tp+fn)) / (tp/((tp+fp))+(tp/(tp+fn))) )

