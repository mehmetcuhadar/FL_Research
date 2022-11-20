import pandas as pd
import numpy as np
from sklearn import linear_model, svm
from sklearn.cluster import KMeans

def read_label_flipping_data():
    df = pd.read_csv("3000_results.csv", header= None)
    return df

def read_backdoor_data():
    df = pd.read_csv("backdoor_results.csv",header= None)
    return df

def read_workers_selected_data():
    df = pd.read_csv("3000_workers_selected.csv", header= None)
    return df

def read_poisoned_workers():
    file1 = open("logs/3000.log", 'r')
    Lines = file1.readlines()
    w_arr = []
    start = 0
    end = 0
    
    for line in Lines:
        if "Poisoning data for workers:" in line:
            for i in range(len(line)):
                if line[i] == "[":
                    start = i
                if line[i] == "]":
                    end = i
            w_arr = line[start+1:end]
    w_arr = w_arr.replace(" ", "").split(',')
    poisoned_workers = []
    for i in range(len(w_arr)):
        poisoned_workers.append(int(w_arr[i]))
    return poisoned_workers

def get_workers_binary_array(workers):
    arr = []
    for t in range(workers.shape[0]):
        label = []
        for i in range(50):
            if i in workers[t]:
                label.append(1)
            else:
                label.append(0)
        arr.append(label)
    return arr

def get_prediction_workers_binary_array():
    arr2 = []
    for i in range(50):
        label = []
        for j in range(50):
            if i == j:
                label.append(1)
            else:
                label.append(0)
        arr2.append(label)
    return arr2

def get_malicious_clients(y_kmeans):
    arr1 = []
    arr2 = []
    for i in range(50):
        if y_kmeans[i] == 0:
            arr1.append(i)
        elif y_kmeans[i] == 1:
            arr2.append(i)
    if len(arr1)>len(arr2):
        return arr2
    else:
        return arr1

if __name__ == '__main__':

    data = read_label_flipping_data()
    workers = np.array(read_workers_selected_data())
    X = get_workers_binary_array(workers)
    target = 0
    y = np.array(data[13 + target])*100  
    lm = svm.SVR(gamma = "auto",kernel = "poly", coef0 = 10)
    model = lm.fit(X, y)
    p_bin_arr = get_prediction_workers_binary_array()
    predictions = lm.predict(p_bin_arr)
    kmeans = KMeans(n_clusters=2)
    y_kmeans = kmeans.fit_predict(predictions.reshape(50,1))
    malicious_clients = get_malicious_clients(y_kmeans)

    print(malicious_clients)