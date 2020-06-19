import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from scipy.stats import multivariate_normal as mn
from sklearn import mixture

dataset = np.loadtxt("P3b_train_data.txt")
testset = np.loadtxt("P3b_test_data.txt")
train = pd.DataFrame(dataset)
test = pd.DataFrame(testset)
sp1 = train[train[1]>0]
sp2 = train[train[1]<0]
sp1t = sp1.iloc[:,0:1].values
sp2t = sp2.iloc[:,0:1].values
clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
clf.fit(sp1t)
clf2 = mixture.GaussianMixture(n_components=2, covariance_type='full')
clf2.fit(sp2t)

res = []

for x in testset:
  a = b = 0;
  for i in range(2):
    a += clf.weights_[i] *mn(clf.means_[i],clf.covariances_[i]).pdf(x[0]) 
  for i in range(2):
    b += clf2.weights_[i] *mn(clf2.means_[i],clf2.covariances_[i]).pdf(x[0])   
  if a > b:
   res.append(1)
  else:
   res.append(-1)
print("EM Bayes:")
print("Class 1:\n", "Mixing Coefficient: ", clf.weights_[0],"Mean:",clf.means_[0],"Variance:",clf.covariances_[0])
print("Mixing Coefficient: ", clf.weights_[1],"Mean:",clf.means_[1],"Variance:",clf.covariances_[1])
print("Class -1:\n", "Mixing Coefficient: ", clf2.weights_[0],"Mean:",clf2.means_[0],"Variance:",clf2.covariances_[0])
print("Mixing Coefficient: ", clf2.weights_[1],"Mean:",clf2.means_[1],"Variance:",clf2.covariances_[1])
print(classification_report(testset[:,1], res, labels=[-1,1]))




m1 = sp1t.mean()
m2 = sp2t.mean()
v1 = sp1t.var()
v2 = sp2t.var()


res = []
for x in testset:
  if mn(m1,v1).pdf(x[:1]) > mn(m2,v2).pdf(x[:1]):
   res.append(1)
  else:
   res.append(-1)
print("MLE Bayes:")
print("Class 1:\n", "mean:", m1, "variance:", v1)
print("Class -1:\n", "mean:", m2, "variance:", v2)
print(classification_report(testset[:,1], res, labels=[-1,1]))


i = len(sp1t)
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(pd.concat([pd.DataFrame(sp1t),pd.DataFrame(sp2t)]),[1]*i + [-1]*i)
predicted_values = neigh.predict(testset[:,0:1])
print("Nearest Neighbour:\n",classification_report(testset[:,1], predicted_values, labels=[-1,1]))




"""
print(clf.means_)
print(clf.covariances_)
print(clf.weights_)
print("\n\n")
print(clf2.means_)
print(clf2.covariances_)
print(clf2.weights_)
"""
