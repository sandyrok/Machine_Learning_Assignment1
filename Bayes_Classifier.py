import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from scipy.stats import multivariate_normal as mn
from sklearn import mixture

dataset = np.loadtxt("P1b_train_data_2D.txt", delimiter=",")
testset = np.loadtxt("P1b_test_data_2D.txt", delimiter=",")

train = pd.DataFrame(dataset)
test = pd.DataFrame(testset)
"""
sp = df.sample(20)
print(test.columns.values)
"""
sp1 = train[train[2]>0]
sp2 = train[train[2]<0]

ns = [100]
for i in ns:
 print("Sample Size: ", i,"\n")
 cl1t = sp1.sample(i).iloc[:,0:2]
 cl2t = sp2.sample(i).iloc[:,0:2]
 cov1 = cl1t.cov()
 cov2 = cl2t.cov()
 m1 = cl1t.mean()
 m2 = cl2t.mean()
 """
 print("Class 1 mean: \n", m1.to_numpy(),"\n")
 print("Class 1 covariance: \n", cov1.to_numpy(),"\n")
 print("class 2 mean: \n", m2.to_numpy(),"\n")
 print("class 2 covariance: \n", cov2.to_numpy(),"\n")
 """
 res = []
 for x in testset:
  if mn(m1,cov1).pdf(x[:2]) > mn(m2,cov2).pdf(x[:2]):
   res.append(1)
  else:
   res.append(-1)
 neigh = KNeighborsClassifier(n_neighbors=1)
 neigh.fit(pd.concat([cl1t,cl2t]),[1]*i + [-1]*i)
 predicted_values = neigh.predict(testset[:,0:2])
 print("Bayes Report:")
 print(classification_report(testset[:,2], res, labels=[-1,1]))
 """
 print("Nearest Neighbour Report:")
 print(classification_report(testset[:,2], predicted_values, labels=[-1,1]))
 print("\n\n")
 """
 clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
 clf.fit(pd.concat([cl1t,cl2t]))
 res = []
 res1 = []
 for x in testset:
  a = mn(clf.means_[0],clf.covariances_[0]).pdf(x[0:2]) 
  b = mn(clf.means_[1],clf.covariances_[1]).pdf(x[0:2])
  if a > b:
   res.append(1)
   res1.append(-1) 
  else:
   res.append(-1)
   res1.append(1)
 print(classification_report(testset[:,2], res, labels=[-1,1]))
 print(classification_report(testset[:,2], res1, labels=[-1,1])) 



"""
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(df[x_columns], df[y_columns])
predictions = knn.predict(test[x_columns])
"""

"""
x = dataset[:,0:2]
y = dataset[:,2]
print(x)
print(y)
"""
