# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 08:35:18 2019

@author: Sebastian Pineda

Code implementing Neural Differential Dictionaries with variable keys

Todo: Resources of squared distance implementation
"""
#importing libraries
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.model_selection import train_test_split
import tf_utils as ut
import matplotlib.pyplot as plt

#defining functions
def sq_distance(A, B):

  row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
  row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

  row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
  row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

  return row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B

def kernel (A,B):
    
    d = sq_distance(A,B)
    o = tf.reciprocal(d+1e-4)
    return o

#reading data
data = load_breast_cancer()

X_data = data.data
target = data.target
target = np.vstack((target, 1-target))
target = target.T

X_train, X_test, y_train, y_test = train_test_split(
    X_data, target, test_size=0.33, random_state=42)

print("Shape of training set:",X_train.shape)
print("Shape of test set:", X_test.shape)

#setting hyperparameters
n_train = X_data.shape[0]
n_input = X_data.shape[1]
n_hidden1 = 10
n_hidden2 = 2
n_output = 2
batch_size = 64
n_iterations = 2000
n_keys_per_class = 10
values = np.repeat(np.eye(n_output), n_keys_per_class ,axis=0)
n_keys= values.shape[0]
list_acc = []

#creating the graph
X = tf.placeholder(tf.float32, shape = (None,  n_input))
Y= tf.placeholder(tf.float32, shape = (None, n_output))
V = tf.placeholder(tf.float32, shape = (n_keys, n_output))

weights = {
        'W1' : tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
        'W2' : tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),
        'out' : tf.Variable(tf.truncated_normal([n_hidden2, n_output], stddev=0.1))}

biases = {
        'b1' : tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
        'b2' : tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
        'out' : tf.Variable(tf.constant(0.1, shape=[n_output]))}

keys = tf.Variable(tf.truncated_normal([n_keys, n_hidden2], stddev=0.1))

layer1 = tf.nn.relu(tf.add(tf.matmul(X, weights['W1']), biases['b1']))
layer2 = tf.matmul(layer1, weights['W2']) + biases['b2']

K = kernel(keys, layer2)
output = tf.matmul(tf.transpose(K), V)
#output = tf.nn.sigmoid(tf.add(tf.matmul(layer2, weights['out']), biases['out']))

#to change: be careful with the use of nn.softmax_cross -> the logits can nont be the output of softmax
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output))
W=0.00
reg1 = tf.reduce_mean(tf.multiply(tf.matmul(kernel(keys, keys),V),(1-V)))
reg2 = tf.reduce_mean(tf.multiply(tf.matmul(kernel(keys, keys),V),V))
reg3 = tf.reduce_mean(kernel(keys, keys))
cost = cross_entropy + W*(reg1-reg2) #+ W*tf.reduce_mean(kernel(keys, keys))  
train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)
correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#running the graph
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# train on mini batches
for i in range(n_iterations):

    minibatches = ut.random_mini_batches(X_train, y_train, batch_size, 1)

    for minibatch in minibatches:
        batch_X, batch_Y = minibatch
        _, ks, l2 = sess.run([train_step, keys, layer2], feed_dict={X: batch_X, Y: batch_Y, V: values})
    
    minibatch_loss, acc = sess.run([cross_entropy, accuracy], feed_dict={X: X_train, Y: y_train, V: values})
    list_acc.append(acc)
    
    if i%10==0:
        minibatch_loss, acc = sess.run([cross_entropy, accuracy], feed_dict={X: X_test, Y: y_test, V: values})
        print("Iteration", str(i), "\t| Loss =", str(minibatch_loss), "\t| Acc =", str(acc))
        
#evaluating model
acc_train, ko ,l2 = sess.run([accuracy, K, layer2], feed_dict={X: X_train, Y: y_train, V: values})
acc_test = sess.run(accuracy, feed_dict={X: X_test, Y: y_test, V: values})

print("Accuracy in train:", acc_train)
print("Accuracy in test:", acc_test)

plt.plot(list_acc)
plt.show()

keys_, y_hat = sess.run([keys, output], feed_dict={X: X_train, Y: y_train, V: values})

import matplotlib.pyplot as plt

color = np.array(["r", "b"])


#plt.plot(l2[np.argmax(y_train,1)==0,0], l2[np.argmax(y_train,1)==0,1], "r.")
#plt.plot(l2[np.argmax(y_train,1)==1,0], l2[np.argmax(y_train,1)==1,1], "b.")

plt.plot(l2[np.argmax(y_hat,1)==0,0], l2[np.argmax(y_hat,1)==0,1], "r.")
plt.plot(l2[np.argmax(y_hat,1)==1,0], l2[np.argmax(y_hat,1)==1,1], "b.")

plt.plot(keys_[np.argmax(values,1)==0,0], keys_[np.argmax(values,1)==0,1], "y*")
plt.plot(keys_[np.argmax(values,1)==1,0], keys_[np.argmax(values,1)==1,1], "g*")
plt.grid()
plt.show()
