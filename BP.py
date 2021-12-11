
from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp
import numpy as np

with open('train.csv', mode='r') as f:
	List_train = []
	for line in f:
		terms = line.strip().split(',')  # 7*N matrix
		List_train.append(terms)
with open('test.csv', mode='r') as f:
	List_test = []
	for line in f:
		terms = line.strip().split(',')  # 7*N matrix
		List_test.append(terms)

# Convert string column to float
def str_2_flo(dataset):
    for row in dataset:
        for j in range(len(dataset[0])):
            row[j] = float(row[j])
    return dataset

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# add constant feature 1 to as the last feature before label
def add_cons_feature(dataset):
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

list_train = str_2_flo(List_train)
list_test = str_2_flo(List_test)
train_data = add_cons_feature(label(list_train))
test_data = add_cons_feature(label(list_test))
train_len = len(train_data)
test_len = len(test_data)
dim_s = len(train_data[0]) - 1

from random import seed
from random import random

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

seed(1)
network = initialize_network(2, 1, 2)
for layer in network:
	print(layer)

def sign_func(x):
	y = 0
	if x > 0:
		y = 1
	else:
		y = -1
	return y


def error_compute(xx, yy):
	cnt = 0
	length = len(xx)
	for i in range(length):
		if xx[i] != yy[i]:
			cnt = cnt + 1
	return cnt / length


#Sigmoid={}
#Gamma={}
#width_m={}
#train_error ={}
#test_error={}

# sigmoid can take in a vector as input
@np.vectorize
def sigmoid(x):
	# avoid overflow
	if x < -100:
		temp = 0
	else:
		temp = 1 / (1 + np.e ** (-x))
	return temp

def ReLU(x):
	return np.maximum(0.0, x)

def phi(x):
	return x * (1.0 - x)

# rate schedule
def gamma(t, gamma_0, d):
	return gamma_0 / (1 + (gamma_0 / d) * t)

# generating Gaussian rv matrix with shape 'size' (tuple)
def rand_weight(shape):
	mean = 0
	std_dev = 1
	return np.random.normal(mean, std_dev, shape)

def zero_weight(shape):
	return np.zeros(shape)

def ones_weight(shape):
	return np.ones(shape)


act_func = sigmoid  # choose activation function
wt_generation = rand_weight  # choose weight initialization method


class NN:
	def __init__(self,
				 num_of_input_nodes,
				 num_of_output_nodes,
				 num_of_hidden_1_nodes,
				 num_of_hidden_2_nodes):
		self.num_of_input_nodes = num_of_input_nodes
		self.num_of_output_nodes = num_of_output_nodes
		self.num_of_hidden_1_nodes = num_of_hidden_1_nodes
		self.num_of_hidden_2_nodes = num_of_hidden_2_nodes
		self.wt_matrix_initial()
		self.stepsize = gamma

	def wt_matrix_initial(self):
		self.wt_mat_layer_0 = wt_generation([self.num_of_hidden_1_nodes - 1, self.num_of_input_nodes])
		self.wt_mat_layer_1 = wt_generation([self.num_of_hidden_2_nodes - 1, self.num_of_hidden_1_nodes])
		self.wt_mat_layer_2 = wt_generation([self.num_of_output_nodes, self.num_of_hidden_2_nodes])

#running the network with an input vector input_vector
	# turn the input vector into a column vector
	def forward_propagate(self, input_vec):
		input_vec = np.array(input_vec, ndmin=2).T
		interm_val_vec_1 = np.dot(self.wt_mat_layer_0, input_vec)
		interm_val_vec_1 = act_func(interm_val_vec_1)  # z values in layer 1
		interm_val_vec_2 = np.dot(self.wt_mat_layer_1, np.concatenate((interm_val_vec_1, [[1]]), axis=0))
		interm_val_vec_2 = act_func(interm_val_vec_2)  # z values in layer 2
		output = np.dot(self.wt_mat_layer_2, np.concatenate((interm_val_vec_2, [[1]]), axis=0))
		return sign_func(output)

	# returns the node z-values at each layer

	def train(self, input_vec, true_label, iteration_cnt, gamma_0, d):
		input_vec = np.array(input_vec, ndmin=2).T  # column array vector
		interm_val_vec_1 = act_func(np.dot(self.wt_mat_layer_0, input_vec))  # col vector
		interm_val_vec_2 = act_func(np.dot(self.wt_mat_layer_1, np.concatenate((interm_val_vec_1, [[1]]), axis=0)))
		output = np.dot(self.wt_mat_layer_2, np.concatenate((interm_val_vec_2, [[1]]), axis=0))
		# calculate the partial derivative matrix
		# gradient W_2
		output_error = output - true_label
		grad_w_2 = output_error * (np.concatenate((interm_val_vec_2, [[1]]), axis=0)).T
		# gradient W_1
		hidden_error_vec_2 = output_error * (self.wt_mat_layer_2[0, :][:-1])  # row vector
		temp = hidden_error_vec_2 * (interm_val_vec_2.T) * (1 - (interm_val_vec_2.T))  # row vector
		tt = np.concatenate((interm_val_vec_1, [[1]]), axis=0)
		grad_w_1 = np.dot(tt, temp).T  # matrix
		# gradient W_0
		alpha_vec = self.wt_mat_layer_2[0, :][:-1]
		beta_vec = phi(interm_val_vec_2.T)  # row vec
		ab = alpha_vec * beta_vec  # row
		tpp = np.zeros((self.num_of_hidden_1_nodes - 1, 1))
		for i in range(self.num_of_hidden_1_nodes - 1):
			tpp[i, 0] = output_error * np.inner(ab, self.wt_mat_layer_1[:, i].T) * phi(interm_val_vec_1.T)[0, i]
		grad_w_0 = np.dot(tpp, input_vec.T)
		# update W_2,W_1, W_0 weights
		self.wt_mat_layer_2 = self.wt_mat_layer_2 - gamma(iteration_cnt, gamma_0, d) * grad_w_2
		self.wt_mat_layer_1 = self.wt_mat_layer_1 - gamma(iteration_cnt, gamma_0, d) * grad_w_1
		self.wt_mat_layer_0 = self.wt_mat_layer_0 - gamma(iteration_cnt, gamma_0, d) * grad_w_0
		iteration_cnt = iteration_cnt + 1
		return iteration_cnt

#Denote M=5,10,25,50,100,gamma=0.01,0.02,0.05,d=1,2
M = 100
gamma_0 = 0.02
d = 2
# create Neural network
nn = NN(num_of_input_nodes=5,
		num_of_output_nodes=1,
		num_of_hidden_1_nodes=M,
		num_of_hidden_2_nodes=M)

# train NN
cnt = 1
for i in range(train_len):
	cnt = nn.train(train_data[i][0:dim_s], train_data[i][-1], cnt, gamma_0, d)
	print(nn.wt_mat_layer_0)

# prediction on training data
pred_seq = []
for i in range(train_len):
	true_labels = [row[-1] for row in train_data]
	pred_seq.append(nn.forward_propagate(train_data[i][0:dim_s]))
print('train error = ', error_compute(pred_seq, true_labels))

# prediction on test data
pred_seq_test = []
for i in range(test_len):
	true_labels_test = [row[-1] for row in test_data]
	pred_seq_test.append(nn.forward_propagate(test_data[i][0:dim_s]))
print('test error = ', error_compute(pred_seq_test, true_labels_test))


#dict_Sigmoid=sigmoid
#dict_Gamma=gamma
#dict_width_m=M


