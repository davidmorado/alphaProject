import numpy as np 
import matplotlib.pyplot as plt 



def s(x, k):
    return 100 / (1 + np.exp(-k*(x-10)))

def f(x, k):
    return (1 / s(1, k)) * s(2*x - 1, k)


x = np.linspace(-10, 500, 10000)

from scipy.stats import norm
mu = 150
var = 100
plt.plot(x,500*norm.cdf((x-mu)/var))
plt.show()

