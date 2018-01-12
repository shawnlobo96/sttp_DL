# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 04:05:58 2018

@author: Neel
"""
import numpy as np
from random import seed
from random import random
from math import exp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#BackPropagation for wheat data set
#%%
def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats
#%%
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0]) 

#%%
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network
 
#seed(1)
#network = initialize_network(2, 1, 2)
#for layer in network:
#	print(layer)
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))
# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    print(' lrate=%.3f' % l_rate)
    SSE=[]
    for epoch in range(n_epoch):
        #SSE=np.append(SSE,sum_error)
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            h=np.int(row[-1])
            expected[h] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
            
        #print(' lrate=%.3f' % l_rate)
        print('>epoch=%d, error=%.3f' % (epoch, sum_error))
        SSE=np.append(SSE,sum_error,axis=None)
    plt.figure(1)
    #print(SSE)     
    plt.plot(SSE)
    plt.ylabel('Sum Square Error')
    plt.xlabel('epoch')
    plt.show()
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']
            
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


#%%
seed(1)
df= pd.read_csv('seeds_dataset.csv')
data=df.iloc[:,:].values
np.random.shuffle(data)
minmax = dataset_minmax(data)
normalize_dataset(data, minmax)
xx=data[:,0:-1]
y=data[:,-1]-1
x_train, x_test, y_train, y_test = train_test_split(xx, y, test_size=0.3)
#%%
L=np.asarray([y_train])
dataset=np.concatenate((x_train,L.T), axis=1)
n_inputs = len(x_train[0])
n_outputs=len(set(y_train))
network = initialize_network(n_inputs, 2, n_outputs)
#%%
train_network(network,dataset, 0.5, 50, n_outputs)

#%%
#testing
L1=np.asarray([y_test])
dataset1=np.concatenate((x_test,L1.T), axis=1)
d1,c1=dataset1.shape
sum_err=0
for row in dataset1:
    prediction = predict(network, row)
    if prediction!=row[-1]:
        sum_err+=1
    print('Expected=%d, Got=%d' % (row[-1], prediction))
r,c=df.shape
#%%
acc=(d1-sum_err)*100/d1
print('\n')
print('accuracy=%.2f' % (acc),'%')