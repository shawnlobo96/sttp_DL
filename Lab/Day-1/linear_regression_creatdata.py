# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:16:44 2017

@author: abc
"""

import numpy as np
import matplotlib.pyplot as plt

N = 100
with open('data_1d.csv', 'w') as f:
    X = np.random.uniform(low=0, high=50, size=N)
    Y = 2*X + 1 + np.random.normal(scale=5, size=N)
    for i in range(N):
        f.write("%s,%s\n" % (X[i], Y[i]))
t = np.arange(0., 100, 1)
plt.close()
plt.plot(t,X,'r',t,Y,'b')  
plt.show()
#%%      
N = 100
w = np.array([2, 3]) # vector of coefficients
with open('data_2d.csv', 'w') as f:
    X = np.random.uniform(low=0, high=100, size=(N,2))
    Y = np.dot(X, w) + 1 + np.random.normal(scale=5, size=N)
    for i in range(N):
        f.write("%s,%s,%s\n" % (X[i,0], X[i,1], Y[i]))
plt.close()
plt.plot(t,X,'r',Y,'b')  
plt.show()
#%%        
N = 100
with open('data_poly.csv', 'w') as f:
    X = np.random.uniform(low=0, high=100, size=N)
    X2 = X*X
    Y = 0.1*X2 + X + 3 + np.random.normal(scale=10, size=N)
    for i in range(N):
        f.write("%s,%s\n" % (X[i], Y[i]))