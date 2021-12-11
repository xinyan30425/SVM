
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

with open('train.csv',mode='r') as f:
    List_train=[]
    for line in f:
        terms=line.strip().split(',') # 7*N matrix
        List_train.append(terms)
with open('test.csv',mode='r') as f:
    List_test = []
    for line in f:
        terms=line.strip().split(',') # 7*N matrix
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
def label(dataset):
    temp = dataset
    for i in range(len(dataset)):
        temp[i][-1] = 2*dataset[i][-1]-1
    return temp
                   
list_train = str_2_flo(List_train)  #convert to float  types data
list_test = str_2_flo(List_test)

train_data = add_cons_feature(label(list_train))
test_data = add_cons_feature(label(list_test))

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

# var-- variance
def loss_fun(w, dataset, var):
    seq = []
    t1 = 1/(2*var)*np.inner(w,w)
    for row in dataset:
        temp = -row[-1]*np.inner(w, row[0:dim_s])
        if temp > 100:
            t2 = temp
        else:
            t2 = math.log(1+ math.e**(temp))
        seq.append(t2)
    sum_ = sum(seq)
    return  sum_ + t1

# returns an array
def sgd_grad(w, sample, var):
     cc = train_len*sample[-1]*(1-sigmoid(sample[-1]*np.inner(w, sample[0:dim_s])))
     return np.asarray([w[i]/var - cc*sample[i] for i in range(dim_s)])
 
def grad(w,var):
    temp = []
    for row in train_data:
        temp.append(row[-1]*(1-sigmoid(row[-1]*np.inner(w, row[0:dim_s]) ))*np.asarray(row[0:dim_s]) ) 
    return w/var - sum(temp) 

def GD(w, var, gamma_0, d):
    w = np.asarray(w)
    loss_seq =[]
    for i in range(train_len):
        w = w - gamma(i, gamma_0, d)*grad(w, var)
        loss_seq.append(loss_fun(w, train_data, var)) 
    return [w, loss_seq]
    

def gamma(t, gamma_0, d): 
    return gamma_0/(1 + (gamma_0/d)*t)

def sgd_single(w, perm, var, iter_cnt, gamma_0, d):
    w = np.asarray(w)
    loss_seq = []
    for i in range(train_len):
        w = w - gamma(iter_cnt, gamma_0, d)*sgd_grad(w, train_data[perm[i]], var)
        loss_seq.append(loss_fun(w, train_data, var)) 
        iter_cnt = iter_cnt + 1
    return [w, loss_seq, iter_cnt]


def sgd_epoch(w, var, T, gamma_0, d):
    iter_cnt = 1
    loss = []
    for i in range(T):
        perm = np.random.permutation(train_len)
        [w, loss_seq, iter_cnt] = sgd_single(w, perm, var, iter_cnt, gamma_0, d)
        loss.extend(loss_seq)
        print('epochs=', i)
    return [w, loss, iter_cnt]

#  MAP learning
def map_main(VV, TT):
    gamma_0 =1
    d =1
    train_err = []
    test_err = []
    for var in VV:
        w = np.zeros(5)
        [wt, loss, cnt] = sgd_epoch(w, var, TT ,gamma_0, d)
        train_err.append(predict(wt, train_data))
        test_err.append(predict(wt, test_data))
    return [train_err, test_err]


V = [0.01, 0.1, 0.5, 1,3,5,10,100]
T = 100
[train_error, test_error] = map_main(V, T)
print('train_error=', train_error)
print('test_error =', test_error)

 

#w =list(np.zeros(len(train_data[0])-1))
# var = 1
# [w, loss, iter_cnt] = sgd_epoch(w, var, T, gamma_0, d)
# plt.plot(loss[0:100])
# =============================================================================






   
    

    

    






