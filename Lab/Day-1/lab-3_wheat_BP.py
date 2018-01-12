# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 12:30:53 2017

@author: abc
"""
#Backpropagation for iris classification
import numpy as np
#import numpy.linalg as ln
import pandas as pd

def sigmoid(u):
  return 1. / (1. + np.e ** -u)

#classes = ['setosa', 'versicolor', 'virginica']
#%%
def predict(X, W01,W12,row):
    g10 = sigmoid(np.dot(X[row,:], W01.T)) # n x m
    g21 = sigmoid(np.dot(g10, W12.T)) # n x 3 
    return outputs.index(max(g21))
    
def back_propagation(X, labels, m, n_classes,regularization=True):
  """Multilayer Neural Network
  input units:  number of columns in X
  output units=n_classes: 3 for iris.csv
  :param X: d x n input matrix (d will be 4 for Iris)
  :param m: number of intermediate units, hidden neurons
  """
  np.random.seed(2)
  d, n = X.shape
  # Bias input
  #X = np.vstack((np.ones(n), X)).T # augumented; n x d+1

  # read label, and convert 3 unit format (001, 010, 100)
  b = -1 * np.ones((n, n_classes))
  for i in range(n):
    #idx = classes.index(labels[i])
    idx=labels[i]-1
    b[i, idx] = 1.

  # weight matrix from input layer (d+1=3) to intermediate layer (m)
  W01 = np.random.randn(m, d)############
  print(W01)
  # weight matrix from intermediate layer (m) to output layer (3)
  W12 = np.random.randn(3, m)

  epoch = 0
  learning_rate = .005
  
  l = .01 # lambda for regularization

  # learning
  while epoch < 2000:
    epoch += 1

    # compute output for n input data
    g10 = sigmoid(np.dot(W01, X)) # n x m
    g21 = sigmoid(np.dot(W12,g10)) # n x 3

    

    # epsilon from output layer to intermediate layer
    # with converting 0, 1 output (g21) to -1, 1 output (same as b)
    e21 = ((g21 * 2 -1) - b.T) * g21 * (1. - g21) # n x 3

    # epsilon from intermediate layer to input layer
    e10 = np.dot(e21, W12) * g10 * (1. - g10) # n x m
#    W12 -= learning_rate * np.dot(e21.T, g10) # 3 x m
#    W01 -= learning_rate * np.dot(e10.T, X) # m x d+1
#    # adjust weights
    if regularization:
      W12 -= learning_rate * (np.dot(e21.T, g10) + (l * W12)) # 3 x m
      W01 -= learning_rate * (np.dot(e10.T, X) + (l * W01)) # m x d+1
    else:
      W12 -= learning_rate * np.dot(e21.T, g10) # 3 x m
      W01 -= learning_rate * np.dot(e10.T, X) # m x d+1

  return W01, W12
  
#%%
X = []
labels = []
df= pd.read_csv('seeds_dataset.csv')
labels = df.iloc[1:-1, -1].values
X=df.iloc[1:-1, 0:-1].values
X = np.asarray(X)
labels = np.asarray(labels)
c,n=X.shape
X1 = np.vstack((np.ones(c), X.T))
#%%
W01, W12 = back_propagation(X1.T, labels, 6,3,True )
#d, n = X.shape
#%%
X1 = np.column_stack((np.ones(d), X))
#%%
g1p = sigmoid(np.dot(X1, W01.T)) # n x m
g2p = sigmoid(np.dot(g1p, W12.T)) # n x 3
#%% 
#predict=g2p.index(max(g2p))
predict1=np.argmax(g2p, axis=1)
#predict=predict1
predict= np.where(predict1 == 0, 'setosa',(np.where(predict1==1,'versicolor','virginica')))
print(predict)
err=0
for jj in range(0,len(labels)):
    if labels[jj]!=predict[jj]:
        err=err+1
print('prediction error=',err)   
    
#%%
