import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as ras

data = pd.read_csv('data-logistic.csv',header=-1)
x1 = data[1].values
x2 = data[2].values
y = data[0].values

def weights(w1,w2,k,c):
    sum1=0
    sum2=0
    w1_pred=-10000
    w2_pred=-10000
    itter = 0
    while (np.sqrt((w1-w1_pred)**2 + (w2-w2_pred)**2) > 1e-5) and (itter < 10000):
        sum1=0
        sum2=0
        w1_pred=w1
        w2_pred=w2
        for i in range (0,205):
            sum1 =sum1+ y[i]*x1[i]*( 1-( 1/( 1+np.exp(-y[i]*(w1*x1[i]+w2*x2[i]) ) )    ))
            sum2 =sum2+ y[i]*x2[i]*( 1-( 1/( 1+np.exp(-y[i]*(w1*x1[i]+w2*x2[i]) ) )    ))

        w1 = w1 + k/205*sum1-k*c*w1
        w2 = w2 + k/205*sum2-k*c*w2
        itter +=1
    print(w1,w2, itter)

    return w1,w2


def sigma(w1,w2,x1,x2):   
    p=1/(   1+np.exp(   -w1*x1[i]-w2*x2[i]  )    )
    return p

pvect=[]
w1,w2 = weights(0,0,0.1,10)                                   
print(w1,w2)

for i in range (205):    
    pvect.append(sigma(w1,w2,x1,x2))
  #  print(sigma(w1,w2,x1,x2))
    
y_true = np.array(y)
y_scores = np.array(pvect)
print(ras(y_true, y_scores),'jopa')    
#print(func(w1,w2,y,x1,x2))
    
