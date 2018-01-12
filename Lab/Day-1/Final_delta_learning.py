# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:21:40 2017

@author: abc
"""
# delta learning
import numpy as np
import matplotlib.pyplot as plt

N = 10
D = 3
X = np.zeros((N, D))
X[:,0] = 1 # bias term
X[:5,1] = 1
X[5:,2] = 1
Y = np.array([0]*5 + [1]*5)
# print X so you know what it looks like
print("X:", X)
#%%
# won't work!
# w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
#Logic gates
#%%
#X=np.array([[1,0,0],
#           [1,0,1],
#           [1,1,0],
#           [1,1,1]])
#Y=np.array([0,0,0,1])
#N,D=X.shape 
#%%
# let's try gradient descent
costs = [] # keep track of squared error cost
np.random.seed(1)
w = np.random.randn(D)# / np.sqrt(D) # randomly initialize w
learning_rate = 0.01
err=[]
for t in range(500):
  # update w
  Yhat = X.dot(w)
  delta = Yhat - Y
  w = w - learning_rate*X.T.dot(delta)

  # find and store the cost
  mse = delta.dot(delta) / N
  #err.append(mse)
  costs.append(mse)

# plot the costs
plt.close()
plt.plot(costs)
plt.title('sum square error')
plt.xlabel('epoch')
plt.show()

print("final w:", w)
Ypred = np.around(X.dot(w))
print("predicted Class:",Ypred.astype(int))
print("target Class:",Y)
