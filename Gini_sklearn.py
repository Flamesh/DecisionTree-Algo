import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
import time

def get_data_to_list(X):
    npoints = X.count()[0]
    list_X = []
    for n in range(npoints):
        x = X.iloc[n, :]
        list_X.append(x)
    return list_X

time_start = time.time()

df = pd.read_csv('datasets/trainning.csv')

X = df.iloc[:, :-1]
y = df.iloc[:,-1]

X = get_data_to_list(X)

dv = DictVectorizer()
dv.fit(X)

X_vec = dv.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size = 0.3)

clf_gini = DecisionTreeClassifier(criterion='gini', random_state=100, max_depth=50, min_samples_leaf=2)
clf_gini.fit(X_train, y_train)

pred = clf_gini.predict(X_test)
print('rate:', metrics.accuracy_score(y_test, pred))

time_end = time.time()
print('time run:', time_end - time_start)
