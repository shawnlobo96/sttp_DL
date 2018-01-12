# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:29:42 2017

@author: abc
"""

# shows how linear regression analysis can be applied to 1-dimensional and 2-dimensional data

from __future__ import print_function, division
from builtins import range
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
#%%

# load the data

X = []
Y = []
for line in open('data_1d.csv'):
    x, y = line.split(',')
    X.append(float(x))
    Y.append(float(y))
#%%
#X = []
#Y = []
#for line in open('data_2d.csv'):
#    x1,x2,y1 = line.split(',')
#    X.append([float(x1),float(x2)])
#    #XX.append(float(x2))
#    Y.append([float(y1)])

#%%
# let's turn X and Y into numpy arrays since that will be useful later
X = np.asarray(X)
Y = np.asarray(Y)

#%%
# let's plot the data to see what it looks like
plt.close()
plt.scatter(X,Y,marker='o',c='r',s=20)
#plt.scatter(X[:,0], Y,marker='^',c='g',s=15)
plt.show()

# will look like a straight line 

#%%
# apply the equations we learned to calculate a and b
# denominator is common

denominator = X.dot(X) - X.mean() * X.sum()
a = ( X.dot(Y) - Y.mean()*X.sum() ) / denominator
b = ( Y.mean() * X.dot(X) - X.mean() * X.dot(Y) ) / denominator

#betaHat = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
# let's calculate the predicted Y
Yhat = a*X + b
#Yhat=np.matmul(X,betaHat)
# let's plot everything together to make sure it worked
t=t = np.arange(0., 100, 1)
plt.close()
#plt.scatter(t, Y)
plt.plot(t, Yhat,'r-',t,Y,'b')
plt.show()

# determine how good the model is by computing the r-squared
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
#r2=1-np.matmul(d1.T,d1)/np.matmul(d2.T,d2)
print("the r-squared is:", r2)