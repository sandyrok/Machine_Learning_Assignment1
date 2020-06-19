import numpy as np
from numpy import linalg as la
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from scipy.stats import multivariate_normal as mn

dataset = np.loadtxt("P2c_train_data_20D.txt", delimiter=",")
testset = np.loadtxt("P2c_test_data_20D.txt", delimiter=",")

train = pd.DataFrame(dataset)
test = pd.DataFrame(testset)
"""
sp = df.sample(20)
print(test.columns.values)
"""
sp1 = train[train[20]>0]
sp2 = train[train[20]<0]
cl1 = sp1.iloc[:,0:20]
cl2 = sp2.iloc[:,0:20]
sm1 = np.array([0]*20)
sm2 = np.array([1]*20)

ns = [50, 100, 300]
for i in ns:
 cl1t = cl1.sample(i)
 cl2t = cl2.sample(i)
 cov1 = cl1t.cov()
 cov2 = cl2t.cov()
 m1 = cl1t.mean()
 m2 = cl2t.mean()
 """
 print(np.linalg.det(cov1))
 print(np.linalg.det(cov2))
 """
 res = []
  
 for x in testset:
  if mn(m1,cov1).pdf(x[:20]) > mn(m2,cov2).pdf(x[:20]):
   res.append(1)
  else:
   res.append(-1)
 print("Sample size:",i)
 print("Bayes Report")
 print(classification_report(testset[:,20], res, labels=[-1,1]))
 neigh = KNeighborsClassifier(n_neighbors=1)
 neigh.fit(pd.concat([cl1t,cl2t]),[1]*i + [-1]*i)
 predicted_values = neigh.predict(testset[:,0:20])
 print("Nearest Neighbour Report:")
 print(classification_report(testset[:,20], res, labels=[-1,1]))



