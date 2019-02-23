import pandas as pd 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np 
from sklearn import metrics
import time
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer 

def get_data_to_list(X):
    npoints = X.count()[0]
    list_X = []
    for n in range(npoints):
        x = X.iloc[n, :]
        list_X.append(x)
    return list_X
time_start = time.time()

df = pd.read_csv('datasets/training_data.csv')
#df_test = pd.read_csv('datasets/test.csv')
X = df.iloc[:, :-1]
y = df.iloc[:,-1]
# attribute = ['dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports', 'attack_cat']
# 43 features
# if dont want using feature, df_train.drop(df_train.columns[[0]], axis=1, inplace=True)
X = get_data_to_list(X)
#print(X.head())
#print(y.head())

#for i in test: print(i)
dv = DictVectorizer()
dv.fit(X)

X_vec = dv.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size = 0.3)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
print('rate:', metrics.accuracy_score(y_test, pred))

time_end = time.time()
print('time run:', time_end - time_start)


'''
clf = RandomForestClassifier(n_estimators=100)

dv = DictVectorizer()
dv.fit(X_train)
X_vec = dv.fit_transform(X_train)
clf.fit(X_vec, y_train)
#y_pred = clf.predict(X_test)


#print('rate :', metrics.accuracy_score(y_test, y_pred))
'''
