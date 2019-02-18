from ID3 import DecisionTreeID3
from C45 import DecisionTreeC45
import time
import pandas as pd 
import numpy as np 
if __name__ == "__main__":
    
    time_start = time.time()

    df_train = pd.read_csv('train.csv')
    
    X_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1]
    
    
    tree = DecisionTreeC45(max_depth = 10, min_samples_split = 2)
    tree.fit(X_train,y_train)
    
    
    df_test = pd.read_csv('test.csv')
    
    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1]
    #print(len(X_test), len(y_test))
    
    predict = tree.predict(X_test)
    predict = list(predict)
    #print(predict)
    y_test = list(y_test)
    #print(predict)
    sum = 0
    for i in range(len(y_test)):
        if(predict[i]==y_test[i]):
            sum+=1

    print(sum/len(predict))
    time_end = time.time()

    print(time_end - time_start)
    
    
    
