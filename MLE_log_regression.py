
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
#========================= data pre-processing =================================
with open('train.csv',mode='r') as f:
    List_train=[]
    for line in f:
        terms=line.strip().split(',')
        List_train.append(terms)
with open('test.csv',mode='r') as f:
    List_test = []
    for line in f:
        terms=line.strip().split(',')
        List_test.append(terms)
        
def str_2_flo(dataset):
    for row in dataset:
        for j in range(len(dataset[0])):
            row[j] = float(row[j])
    return dataset

def add_cons_feature(dataset):   # add constant feature 1 to as the last feature before label
    label = [row[-1] for row in dataset]
    temp = dataset
    for i in range(len(dataset)):
        temp[i][-1] = 1.0
    for i in range(len(dataset)):
        temp[i].append(label[i])
    return temp

# convert label {0,1} to {-1,1}
def polar_label(dataset):
    temp = dataset
    for i in range(len(dataset)):
        temp[i][-1] = 2*dataset[i][-1]-1
    return temp
                   
list_train = str_2_flo(List_train)
list_test = str_2_flo(List_test)

train_data = add_cons_feature(polar_label(list_train))
test_data = add_cons_feature(polar_label(list_test))

train_len = len(train_data)
test_len = len(test_data)
dim_s = len(train_data[0]) -1
def sign_func(x):
    y = 0
    if x> 0:
        y = 1
    else:
        y=-1
    return y
def error_compute(xx,yy):
    cnt = 0
    length =len(xx)
    for i in range(length):
        if xx[i]!= yy[i]:
            cnt = cnt + 1
    return cnt/length

# returns error rate
def predict(wt, dataset):
    pred_seq =[];
    for i in range(len(dataset)):
        pred_seq.append(sign_func(np.inner(dataset[i][0:len(dataset[0])-1], wt)))
    label = [row[-1] for row in dataset]
    return error_compute(pred_seq, label) 

def sigmoid(x):
    if x < -100:
        temp = 0
    else:
        temp = 1/(1 + math.e**(-x))
    return temp


def loss_fun(w, dataset):
    seq = []

    for row in dataset:
        temp = -row[-1]*np.inner(w, row[0:dim_s])
        if temp > 100:
            t2 = temp
        else:
            t2 = math.log(1+ math.e**(temp))
        seq.append(t2)
    return sum(seq)


def sgd_grad(w, sample):
     cc = train_len*sample[-1]*(1-sigmoid(sample[-1]*np.inner(w, sample[0:dim_s])))
     return np.asarray([-cc*sample[i] for i in range(dim_s)])
    

def gamma(t, gamma_0, d): 
    return gamma_0/(1 + (gamma_0/d)*t)


def sgd_single(w, perm, iter_cnt, gamma_0, d):
    w = np.asarray(w)
    loss_seq = []   # sequence of loss fucntion values
    for i in range(train_len):
        w = w - gamma(iter_cnt, gamma_0, d)*sgd_grad(w, train_data[perm[i]])
        loss_seq.append(loss_fun(w, train_data)) 
        iter_cnt = iter_cnt + 1
    return [w, loss_seq, iter_cnt]


# T-- # of epochs
def sgd_epoch(w, T, gamma_0, d):
    iter_cnt = 1
    loss = []
    for i in range(T):
        perm = np.random.permutation(train_len)
        [w, loss_seq, iter_cnt] = sgd_single(w, perm, iter_cnt, gamma_0, d)
        loss.extend(loss_seq)
        print('epochs=', i)
    return [w, loss, iter_cnt]

w= np.zeros(5)
T= 100
gamma_0 =1
d =1
[wt, loss, tt] = sgd_epoch(w, T, gamma_0, d)
plt.plot(loss)
plt.xlabel('iterations')
plt.ylabel('empirical loss')
plt.title('T= 1')
plt.show()
print('train error=',predict(wt, train_data))
print('test error=',predict(wt, test_data))






   
    

    

    






