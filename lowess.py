# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np                       #library numpy
import pandas as pd                      #library pandas
import re                                #library for regex functions
from matplotlib import pyplot as plt     #library matplot
from math import ceil
import numpy as np
from scipy import linalg

def lowess(x, y, f, iter=3): #=2. / 3.
   
    # f is smoothing span - large span is smoother function 
    
    """lowess(x, y, f=2./3., iter=3) -> yest
    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations.
    """
    n = len(x)
    #r = int(ceil(f * n))
    kk=[np.abs(x - x[i])[1] for i in range];kk
    r = 5 # x[i]'ye en yakın r'th noktayo buluyor
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)] #Ona en yakın r'th noktayı buluyor
    print(h)
    '''>>> a = np.arange(10)
       >>> a
       array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
       >>> np.clip(a, 3, 6, out=a)
       array([3, 3, 3, 3, 4, 5, 6, 6, 6, 6])
    '''
    #values outside of h becomes 0 inside h becomes 1
    
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0);w
    np.shape(w)
    #a=np.array([3,2,1]);a
    
    w = (1 - w ** 3) ** 3;w
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2

    return yest

if __name__ == '__main__':
    import math
    n = 100
    x = np.linspace(0, 2 * math.pi, n);x
    
    y = np.sin(x) + 0.3 * np.random.randn(n);y
    
    x=np.array([1,4,5,3,5])
    y=np.array([3,3,2,1,4])
    
    x=np.arange(10);x#type(x)
    #a=asarray(x);a
    y = np.sin(x);y
    
    f = 0.25
    yest = lowess(x, y, f=f, iter=3)

    import pylab as pl
    pl.clf()
    pl.plot(x, y, label='y noisy')
    pl.plot(x, yest, label='y pred')
    pl.legend()
    pl.show()

x=np.array([1,4,5,3,5])
y=np.array([3,3,2,1,4])    
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x,y,'.')    


import tensorflow as tf

# Initialize two constants
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

# Multiply
result = tf.multiply(x1, x2)

# Print the result
print(result)    



from math import ceil
import numpy as np
from scipy import linalg


def lowess(x, y, f=2. / 3., iter=3):
    """lowess(x, y, f=2./3., iter=3) -> yest
    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations.
    """
    n = len(x);n
    r = int(ceil(f * n));r
    r=5
    kk=[np.abs(x - x[i]) for i in range(n)] ;print(kk)
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2

    return yest

if __name__ == '__main__':
    import math
    n = 100
    x = np.linspace(0, 2 * math.pi, n);print(x);len(x)
    y = np.sin(x) + 0.3 * np.random.randn(n);print(y)
    
    '''x=np.array([1,4,5,3,5])
    y=np.array([3,3,2,1,4])
    
    x=np.random.randn(100);print(x)
    y = np.sin(x) ;print(y)'''
    
    f = 0.25
    yest = lowess(x, y, f=f, iter=3)

    import pylab as pl
    pl.clf()
    pl.plot(x, y, label='y noisy')
    pl.plot(x, yest, label='y pred')
    pl.legend()
    pl.show()