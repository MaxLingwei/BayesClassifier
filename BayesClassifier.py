import os
import csv
import numpy as np
from scipy import stats

def loadCSVdata(filename, tag, data_pos, data_neg):
    data_tmp = []
    f = open(filename, 'r')
    reader = csv.reader(f)
    rows = [row for row in reader]
    f.close()
    tag.append(rows[0])
    for i in range(1, len(rows)):
        label = rows[i][len(rows[i]) - 1]
        if int(label) == 0:
            data_neg.append(rows[i])
        else:
            data_pos.append(rows[i])

    
def predict(data, mean, var, prior):
    prob = list(prior)
    print prob
    #prob = [1, 1]
    for i in range(0, len(data)):
        prob[0] *= stats.norm.pdf(float(data[i]), mean[0][i], var[0][i])
    
    for i in range(0, len(data)):
        prob[1] *= stats.norm.pdf(float(data[i]), mean[1][i], var[1][i])
    
    if prob[0] > prob[1]:
        return 0
    else:
        return 1

def calErrorRate(data, mean, var, prior):
    error = 0
    for i in range(0, len(data)):
        label = data[i][len(data[i]) - 1] 
        if int(label) ^ predict(data[i][:len(data[i]) - 1], mean, var, prior):
            error += 1
        else:
            continue
    return error

if __name__ == '__main__':
    trainfile = './train.csv'
    testfile = './test.csv'
    tag = []
    data_pos = []
    data_neg = []
    mean = []
    sigma = []
    test_pos = []
    test_neg = []
    prior = [0, 0]

    error = 0

    loadCSVdata(trainfile, tag, data_pos, data_neg)
    prior[1] = float(len(data_pos)) / (len(data_pos) + len(data_neg))
    prior[0] = float(len(data_neg)) / (len(data_pos) + len(data_neg))


    mat_tmp = np.mat(data_pos)
    mat_pos = mat_tmp.astype(float)
    mat_tmp = np.mat(data_neg)
    mat_neg = mat_tmp.astype(float)

    mean_tmp = np.mean(mat_neg, axis = 0)
    list_tmp = mean_tmp.tolist()
    mean.append(list_tmp[0])

    mean_tmp = np.mean(mat_pos, axis = 0)
    list_tmp = mean_tmp.tolist()
    mean.append(list_tmp[0])

    sigma_tmp = np.var(mat_neg, axis = 0)
    sigma_tmp = np.sqrt(sigma_tmp)
    list_tmp = sigma_tmp.tolist()
    sigma.append(list_tmp[0])
    
    sigma_tmp = np.var(mat_pos, axis = 0)
    sigma_tmp = np.sqrt(sigma_tmp)
    list_tmp = sigma_tmp.tolist()
    sigma.append(list_tmp[0])
    
    loadCSVdata(testfile, tag, test_pos, test_neg)
    error += calErrorRate(test_pos, mean, sigma, prior)
    error += calErrorRate(test_neg, mean, sigma, prior)

    error_rate = float(error) / float(len(test_pos) + len(test_neg))
    print error_rate
