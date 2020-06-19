import numpy as np
import csv
import re
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

with open("sentiment_analysis.csv", 'r') as file:
 reviews = list(csv.reader(file))
del reviews[0]
df = pd.DataFrame(reviews)
X_train, X_test, y_traint, y_testt = train_test_split(df.iloc[:][1], df.iloc[:][0], test_size=0.2)
y_train = [ 1 if i=="Pos" else -1 for i in y_traint]
y_test  = [ 1 if i=="Pos" else -1 for i in y_testt]

vectorizer = CountVectorizer(stop_words='english')
train_features = vectorizer.fit_transform(X_train)
test_features = vectorizer.transform(X_test)
 
nb = MultinomialNB()
nb.fit(train_features, y_train)
predictions = nb.predict(test_features)
print("Binary Features:")
print(classification_report(y_test, predictions, labels=[-1,1]))
clf = BernoulliNB()
clf.fit(train_features, y_train)
predicts = clf.predict(test_features)
print("TF-IDF")
print(classification_report(y_test, predicts, labels=[-1,1]))
