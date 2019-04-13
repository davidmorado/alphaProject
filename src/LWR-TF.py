#Based on code from https://xavierbourretsicotte.github.io/loess.html

from math import ceil
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import tensorflow as tf

#defining functions
def sq_distance(A, B):

  row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
  row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

  row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
  row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

  return row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B

def kernel (A,B, tau=0.1):
    dist = sq_distance(A,B)
    return(tf.exp( - dist/(2*tau)   ))

n_input = 1
n_output = 1

X = tf.placeholder(tf.float32, shape = (None,  n_input))
Y= tf.placeholder(tf.float32, shape = (None, n_output))
tau = tf.placeholder(tf.float32, shape = ())
W = tf.Variable(tf.truncated_normal([n_input, n_output], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[n_output]))

X_test = tf.placeholder(tf.float32, shape = (None,  n_input))

K = kernel(X, X_test, tau)
y_pred = tf.add(tf.multiply(X, W), b) 

# Mean Squared Error Cost Function 
res = tf.pow(y_pred-Y, 2)
cost =tf.reduce_sum(tf.multiply(res,K))
#cost = tf.reduce_sum(tf.multiply(tf.transpose(K),tf.pow(y_pred-Y, 2)))
#cost = tf.reduce_sum(tf.pow(y_pred-Y, 2))

  
# Gradient Descent Optimizer 
learning_rate = 0.0005
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) 
  
#Initializing noisy non linear data
x = np.linspace(0,1,100).astype(float)
noise = np.random.normal(loc = 0, scale = .25, size = 100)
y = np.sin(x * 1.5 * np.pi ) 
y_noise = y + noise

#running the graph
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

x_test = np.array([0.1])

n_iterations = 100

for i in range(n_iterations):

    k, c, _ = sess.run([K, cost, optimizer ], feed_dict={X:x.reshape([-1,1]), X_test:x_test.reshape([-1,1]), Y:y_noise.reshape([-1,1]), tau:0.001})
    print(c)

y_pred_ = sess.run(y_pred, feed_dict={X:x_test.reshape([-1,1])})                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
y_true = np.sin(x_test * 1.5 * np.pi ) 
       
print("y pred:", y_pred_)
print("y true:", y_true)
                   
def pred_wlr(X_train, X_to_pred, optimizer, y_pred, n_iterations=100, t=0.001):
    
    y_pred_list = []
    
    for i in range(X_to_pred.shape[0]):
        x_test = np.array([X_to_pred[i,]])
        print(x_test)
        for j in range(n_iterations):
            k, c, _ = sess.run([K, cost, optimizer ], feed_dict={X:X_train.reshape([-1,1]), X_test:x_test.reshape([-1,1]), Y:y_noise.reshape([-1,1]), tau:t})
            
        y_pred_ = sess.run(y_pred, feed_dict={X:x_test.reshape([-1,1])})  
        y_pred_list.append(y_pred_[0][0])
    
    return np.array(y_pred_list)

def rmse(y, y_pred):
    
    out = np.sum((y-y_pred)**2)
    return out

rmse_list = []
t_list = [0.0001, 0.001, 0.005, 0.05, 0.5, 1, 2]

for t_ in t_list:
      
    y_pred_ = pred_wlr(x,x, optimizer, y_pred, n_iterations=100, t=t_)
    y_true = np.sin(x * 1.5 * np.pi ) 
    r = rmse(y_pred_, y_true)
    rmse_list.append(r)

plt.plot(rmse_list)
plt.grid()
  
t_over = t_list[0]
t_perf = t_list[3]
t_under = t_list[6]

y_pred_over = pred_wlr(x,x, optimizer, y_pred, n_iterations=100, t=t_over)
y_pred_perf = pred_wlr(x,x, optimizer, y_pred, n_iterations=100, t=t_perf)
y_pred_under = pred_wlr(x,x, optimizer, y_pred, n_iterations=100, t=t_under)

    
plt.plot(y_pred_over, c="r")
plt.plot(y_pred_perf, c="b")
plt.plot(y_pred_under, c="y")
plt.plot(y_true, c="g")
plt.grid()
plt.legend(["Prediction with overfitting",  "Prediction just right", "Prediction with underfitting",  "Ground Truth"])