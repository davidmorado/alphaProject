import pickle
from keras.utils import to_categorical
import sys
import os
import tensorflow as tf
import numpy as np
import math
from keras.models import Model
from keras.datasets import cifar10
import keras
import random
import ast
from data_loader import get_dataset


#defining functions
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[ permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch_Y = shuffled_Y[ k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
  
  
def sq_distance(A, B):

  row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
  row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

  row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
  row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

  return row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B

def kernel (A,B):
    
    d = sq_distance(A,B)
    o = tf.reciprocal(d+bandwidth)
    #o = tf.exp(tf.matmul(B,tf.transpose(A)))
    return o

      
def kernel_gauss( A,B):
    d = sq_distance(A,B)
    o = tf.exp(-(d)/bandwidth)
    return o
  
def conv_net(x, keep_prob, embedding_dim):

    conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 32], mean=0, stddev=0.1))
    conv12_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 32], mean=0, stddev=0.1))

    conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 64], mean=0, stddev=0.1))
    conv22_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], mean=0, stddev=0.1))

    conv1 = tf.nn.conv2d(x, conv1_filter, strides=[1,1,1,1], padding='SAME', name="conv1")
    conv1 = tf.nn.relu(conv1)
    conv12 = tf.nn.conv2d(conv1, conv12_filter, strides=[1,1,1,1], padding='VALID', name="conv12")
    conv12 = tf.nn.relu(conv12)
    conv1_pool = tf.nn.max_pool(conv12, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    conv1_bn = tf.nn.dropout(conv1_pool, 0.75)

    conv2 = tf.nn.conv2d(conv1_bn, conv2_filter, strides=[1,1,1,1], padding='SAME', name="conv2")
    conv2 = tf.nn.relu(conv2)
    conv22 = tf.nn.conv2d(conv2, conv22_filter, strides=[1,1,1,1], padding='VALID', name="conv22")
    conv22 = tf.nn.relu(conv22)
    conv2_pool = tf.nn.max_pool(conv22, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    conv2_bn = tf.nn.dropout(conv2_pool, 0.75)
  
    flat = tf.contrib.layers.flatten(conv2_bn)  

    full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=512, activation_fn=tf.nn.relu)
    full1 = tf.nn.dropout(full1, 0.5)
   
    full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=embedding_dim, activation_fn=tf.nn.relu)
   
    out = full2

    return out









#information from https://towardsdatascience.com/cifar-10-image-classification-in-tensorflow-5b501f7dc77c

# default hyperparameters of the model
epochs = 100
batch_size = 64
keep_probability = 0.25 #dropout parameter
lr = 0.0001
embedding_dim = 100
n_output = num_classes= 10
num_keys_per_class= 1000
bandwidth = 0.01 #bandwidth
dataset = 'cifar10'
split_ratio = 0.1
update_period = 10
nearest_neighbors = 50

# read hyperparameters from command line arguments and overwrite default ones
hp_dict_str = sys.argv[1]
import yaml
hp_dict = yaml.load(hp_dict_str)

#hp_dict = ast.literal_eval(hp_dict_str)
for key,val in hp_dict.items():
    exec(key + '=val')
print('nearest_neighbors: ',nearest_neighbors)
print('update_period: ',update_period)


#(x_train, y_train), (x_val, y_val) = cifar10.load_data()
x_train, x_val, x_test, y_train, y_val, y_test = get_dataset(dataset, split_ratio)


print("Train size:", x_train.shape)
print("Test size:", x_val.shape)




tf.reset_default_graph()

values = np.vstack([np.eye(num_classes)]*num_keys_per_class)
n_keys= values.shape[0]

#initializing placeholders
x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')
y =  tf.placeholder(tf.float32, shape=(None, 10), name='output_y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
V = tf.placeholder(tf.float32, shape = (n_keys, n_output))
keys = tf.placeholder(tf.float32, shape=(None, embedding_dim))
weights = tf.Variable(tf.random.uniform(shape=[n_keys,1], maxval=1))



  

#keys = tf.Variable(tf.truncated_normal([n_keys, embedding_dim],mean=0, stddev=0.1), "keys")
#using the whole training set as keys
embedding = conv_net(x, keep_prob, embedding_dim)
K = kernel(keys, embedding)

dist , indices = tf.nn.top_k(tf.transpose(K), k=nearest_neighbors) #calculating nearest neighbours (kNN)
hit_keys = tf.nn.embedding_lookup(keys, indices) #accessing the vectors of the kNN
V_weighted = tf.multiply(V, tf.tile(weights, [1, num_classes]))
hit_values = tf.nn.embedding_lookup(V_weighted, indices) #accessing the values of the kNN
KV = tf.einsum('ikj,ik->ij', hit_values, dist) #
KV_ = tf.diag(tf.reshape(tf.reciprocal( tf.matmul(KV,tf.ones((n_output,1)))) , [-1]))
output = tf.matmul(KV_, KV)

cross_entropy = tf.reduce_mean(keras.losses.categorical_crossentropy(y, output))
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    # Initializing the variables
   
    minibatches = random_mini_batches(x_train, y_train, 2, 1)

    batch_X, batch_Y = minibatches[0]
    keys_idx = [x for x in range(x_train.shape[0])]
    random.shuffle(keys_idx)
    keys_idx = keys_idx[:n_keys]
    
    sess.run(tf.global_variables_initializer())
    keys_ = sess.run(embedding, feed_dict={x: x_train[keys_idx], keep_prob: keep_probability }) #limiting to 1000 because of RAM
    values = y_train[keys_idx]
    hk, hv, dist_, out, out_, output_, K_ = sess.run([hit_keys, hit_values, dist, V_weighted, KV_, output, K] , feed_dict={x: batch_X, 
                                               y: batch_Y, 
                                               V: y_train[:n_keys],
                                               keep_prob: keep_probability,
                                               keys : keys_})









import os
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
        
        
#tf.reset_default_graph()
grads = tf.train.AdamOptimizer(lr).compute_gradients(loss=cross_entropy)

for grad in grads:
    with tf.name_scope('grads'):
        variable_summaries(grad)
        
with tf.name_scope('performance'):
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('cost', cross_entropy)

with tf.name_scope('weights'):
        variable_summaries(weights)










values = values.astype(float)
acc_train_list = []
acc_validation_list = []

with tf.Session() as sess:
  
  summ_writer_train = tf.summary.FileWriter(os.path.join('./tb_logs','train'), sess.graph)
  summ_writer_test = tf.summary.FileWriter(os.path.join('./tb_logs','test'), sess.graph)

  merged = tf.summary.merge_all()
  
  sess.run(tf.global_variables_initializer())

  for i in range(epochs):

      minibatches = random_mini_batches(x_train, y_train, batch_size, 1)
      acc_temp = []
      
      
      for j, minibatch in enumerate(minibatches):
          
          #if(j%50==0):
            #keys_ = sess.run( embedding, feed_dict={x: x_train[keys_idx], keep_prob: keep_probability})
            #summ_test = sess.run(merged , feed_dict={x: x_val, y: y_val, V: values, keep_prob: keep_probability, keys : keys_})
            #summ_writer_test.add_summary(summ_test, i)
            #summ_writer_test.flush()
            
          batch_X, batch_Y = minibatch
          _, ks = sess.run([train_step, keys], feed_dict={x: batch_X, y: batch_Y, V: values,  keep_prob: keep_probability, keys : keys_})
          minibatch_loss, acc = sess.run([ cross_entropy, accuracy], feed_dict={x: batch_X, y: batch_Y, V: values, keep_prob: keep_probability, keys : keys_})
          
          #print("Acc:", acc, " Loss:", minibatch_loss)
          acc_temp.append(acc)
          
      acc_train_list.append(np.mean(acc_temp))
      #print("Acc:", acc, " Loss:", minibatch_loss)
      #minibatch_loss, acc = sess.run([cross_entropy, accuracy], feed_dict={x: x_train, y: y_train, V: values, keep_prob: keep_probability, keys : keys_})
      #list_acc.append(acc)
      
      print("Train accuracy:", np.mean(acc_temp))
      minibatch_loss, acc, w = sess.run([ cross_entropy, accuracy, weights], feed_dict={x: x_val, y: y_val, V: values, keep_prob: keep_probability, keys : keys_})
      print("Iteration", str(i), "\t| Loss test=", str(minibatch_loss), "\t| Acc test=", str(acc))
      acc_validation_list.append(acc)
      
      if i%update_period==0:
          keys_ = sess.run(embedding, feed_dict={x: x_train[keys_idx], keep_prob: keep_probability })
      
      if i%1==0:
          
          summ_test, minibatch_loss, acc, w = sess.run([merged, cross_entropy, accuracy, weights], feed_dict={x: x_val, y: y_val, V: values, keep_prob: keep_probability, keys : keys_})
          summ_writer_test.add_summary(summ_test, i)
          summ_writer_test.flush()



modelpath = '&'.join([F"{param}={value}" for param, value in hp_dict.items()])

out_results = (hp_dict, acc_train_list, acc_validation_list)
filename = F"gridresults/{modelpath}.pkl"
with open(filename, 'wb') as f:
  pickle.dump(out_results, f)

import matplotlib.pyplot as plt
plt.plot(acc_validation_list)
plt.plot(acc_train_list)
plt.savefig(F'plots/{modelpath}.png')

