# -*- coding: utf-8 -*-
"""cluster_proto_cnn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Tao6YJM-Vm4vzLPI602z4lUbqrBLW_0N

# Data Load
"""

import keras
#from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
import glob
import pickle
from keras.datasets import cifar10

def get_dataset(ds_name, normalize,ratio):
                
        x_train, x_val, x_test, y_train, y_val, y_test = get_cifar10_proto(ratio)
        normalize = True

        if normalize:
            x_train, x_val, x_test = normalize_data(ds_name, x_train, x_val, x_test)
    
        return x_train, x_val, x_test, y_train, y_val, y_test  

def train_val_test_splitter(X, y, ratio, random_state=999):
      x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=999)
      x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=ratio/(1-ratio), random_state=999)
    
      return x_train, x_val, x_test, y_train, y_val, y_test


def normalize_data(ds_name, x_train, x_val, x_test):
      if ds_name == 'cifar10' or ds_name == 'cifar10_proto':
            x_train = x_train/255
            x_val = x_val/255
            x_test = x_test/255
      return x_train, x_val, x_test

def get_cifar10_proto(ratio): 
      (x_train, y_train), (x_test, y_test) = cifar10.load_data()
      X = np.concatenate((x_train, x_test), axis=0)
      y = np.concatenate((y_train, y_test), axis=0)
      
      x_train = np.zeros([10, int(len(X)*(1-2*ratio)/10),32,32,3], dtype=np.float32)
      x_val = np.zeros([10, int(len(X)*(ratio)/10),32,32,3], dtype=np.float32)
      x_test = np.zeros([10, int(len(X)*(ratio)/10),32,32,3], dtype=np.float32)

      for cl in np.sort(np.unique(y)):
          x_train[cl] = X[np.where(y.T[0]==cl)[0][:int(len(X)*(1-2*ratio)/10)]]
          x_val[cl] = X[np.where(y.T[0]==cl)[0][int(len(X)*(1-2*ratio)/10):int(len(X)*(1-ratio)/10)]]
          x_test[cl] = X[np.where(y.T[0]==cl)[0][int(len(X)*(1-ratio)/10):]]
                     
      y_train = [i for i in range(10)]
      y_val = [i for i in range(10)]
      y_test = [i for i in range(10)]
               
      return x_train, x_val, x_test, y_train, y_val, y_test

#Run this to get train-val-test sets (For prototypical: cifar10_proto, omniglot_proto)
x_train, x_val, x_test, y_train, y_val, y_test = get_dataset('cifar10_proto',False,0.2)

x_train.shape, x_val.shape, x_test.shape
#y_train.shape, y_val.shape, y_test.shape
#len(y_train), len(y_val), len(y_test)


def percentage_splitter(train, val, yt ,yv ,merging ,random_selection ,ratio):
    
    if merging == True:

        if len(train.shape)==5:   #Prototypical
            train = np.concatenate((train, val), axis=1)
            if random_selection == True:
                train = shuffle(train, random_state=999)
                train2 = train[:,:int(np.ceil(train.shape[1]*ratio)),:,:,:]
            else:
                train2 = train[:,:int(np.ceil(train.shape[1]*ratio)),:,:,:]
            yt2=yt
            
        elif len(train.shape)==4: #Our Model
            train = np.concatenate((train, val), axis=0)
            yt = np.concatenate((yt, yv), axis=0)
            if random_selection == True:
                print("burasi")
                train, yt = shuffle(train, yt, random_state=999)
                train2 = train[:int(np.ceil(train.shape[0]*ratio)),:,:,:]
                yt2 = yt[:int(np.ceil(yt.shape[0]*ratio)),:]
            else:
                train2 = train[:int(np.ceil(train.shape[0]*ratio)),:,:,:]         
                yt2 = yt[:int(np.ceil(yt.shape[0]*ratio)),:]
            
    else:
        if len(train.shape)==5:   #Prototypical
            if random_selection == True:
                train = shuffle(train, random_state=999)
                train2 = train[:,:int(np.ceil(train.shape[1]*ratio)),:,:,:]
            else:
                train2 = train[:,:int(np.ceil(train.shape[1]*ratio)),:,:,:]
            yt2=yt
        elif len(train.shape)==4:  #Our Model
            if random_selection == True:
                train, yt = shuffle(train, yt, random_state=999)
                train2 = train[:int(np.ceil(train.shape[0]*ratio)),:,:,:]
                yt2 = yt[:int(np.ceil(yt.shape[0]*ratio)),:]
            else:
                train2 = train[:int(np.ceil(train.shape[0]*ratio)),:,:,:]
                yt = np.concatenate((yt, yv), axis=0)
                yt2 = yt[:int(np.ceil(yt.shape[0]*ratio)),:]
    return train2 , yt2


#Merging: merges train and validation
#Random_selection: makes train data shuffle before split so it would select different instances 
#Ratio: selects instances with the given percentage [0-1]
x_train2, y_train2 = percentage_splitter(x_train, x_val, y_train, y_val, merging=True, random_selection=False, ratio=1) 
x_train=x_train2; y_train=y_train2
x_train.shape, x_val.shape, x_test.shape
#y_train.shape, y_val.shape, y_test.shape

#For omniglot-prototypical
train_dataset = x_train
train_classes = y_train
val_dataset = x_val
val_classes = y_val
test_dataset = x_test
test_classes = y_test
n_classes = len(train_classes)
n_val_classes = len(val_classes)
n_test_classes = len(test_classes)

train_dataset.shape, test_dataset.shape

"""# Main Code"""

def conv_net(x, embedding_dim , keep_prob=None):

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
        #conv1_bn = tf.layers.batch_normalization(conv1_pool)

        conv2 = tf.nn.conv2d(conv1_bn, conv2_filter, strides=[1,1,1,1], padding='SAME', name="conv2")
        conv2 = tf.nn.relu(conv2)
        conv22 = tf.nn.conv2d(conv2, conv22_filter, strides=[1,1,1,1], padding='VALID', name="conv22")
        conv22 = tf.nn.relu(conv22)
        conv2_pool = tf.nn.max_pool(conv22, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
        conv2_bn = tf.nn.dropout(conv2_pool, 0.75)
        #conv2_bn = tf.layers.batch_normalization(conv2_pool)

        flat = tf.contrib.layers.flatten(conv2_bn)  

        full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=512, activation_fn=tf.nn.relu)
        full1 = tf.nn.dropout(full1, 0.5)
        #full1 = tf.layers.batch_normalization(full1)

        full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=embedding_dim, activation_fn=tf.nn.relu)
        #full2 = tf.nn.dropout(full2, keep_prob)
        #full2 = tf.layers.batch_normalization(full2)        


        out = full2

        return out
      
def euclidean_distance(a, b):
    # a.shape = N x D
    # b.shape = M x D
    N, D = tf.shape(a)[0], tf.shape(a)[1]
    M = tf.shape(b)[0]
    a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
    b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))
    return tf.reduce_mean(tf.square(a - b), axis=2)

"""## Config Setting"""

n_epochs = 41
n_episodes = 100
n_way = 10
n_shot = 200
n_query = 100
im_width, im_height, channels = 32, 32, 3
h_dim = 64
z_dim = 100
embedding_dim=64

n_classes = 10

values = {
    "embedding_dim": [64],
    "n_way": [10],
    "n_shot": [200],
    "n_query": [100]
}

from sklearn.model_selection import ParameterGrid

cross_combinations = ParameterGrid(values)

"""## Md"""

tf.reset_default_graph()
x = tf.placeholder(tf.float32, [None, None, im_height, im_width, channels])
q = tf.placeholder(tf.float32, [None, None, im_height, im_width, channels])
x_shape = tf.shape(x)
q_shape = tf.shape(q)
num_classes, num_support = x_shape[0], x_shape[1]
num_queries = q_shape[1]
y = tf.placeholder(tf.int64, [None, None])
y_one_hot = tf.one_hot(y, depth=num_classes)
emb_x = conv_net(tf.reshape(x, [num_classes * num_support, im_height, im_width, channels]), embedding_dim)
emb_dim = tf.shape(emb_x)[-1]
emb_x = tf.reduce_mean(tf.reshape(emb_x, [num_classes, num_support, emb_dim]), axis=1)
emb_q = conv_net(tf.reshape(q, [num_classes * num_queries, im_height, im_width, channels]), embedding_dim)
dists = euclidean_distance(emb_q, emb_x)
log_p_y = tf.reshape(tf.nn.log_softmax(-dists), [num_classes, num_queries, -1])
ce_loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_one_hot, log_p_y), axis=-1), [-1]))
acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(log_p_y, axis=-1), y)))

train_op = tf.train.AdamOptimizer().minimize(ce_loss)

sess = tf.InteractiveSession()
init_op = tf.global_variables_initializer()
sess.run(init_op)

print(train_dataset.shape)
n_examples=train_dataset.shape[1];print(n_examples)
print(test_dataset.shape)
n_examples_test = test_dataset.shape[1];print(n_examples_test)

"""## Training"""

filename = '6. proto_cifar10_100_cnn_graph.pkl'

n_epochs = 801
n_episodes = 100
n_way = 10
n_shot = 200
n_query = 100
im_width, im_height, channels = 32, 32, 3
h_dim = 64
z_dim = 100
embedding_dim=64

n_classes = 10

n_test_epochs=5
n_test_episodes = 100
n_test_way = 10
n_test_shot = 200
n_test_query = 100

log_performance = []
convnet_activate = True

for indx, hyper_parameters in enumerate(cross_combinations):
  print('---------------------------------------------------------------------------')
  print('Combination '+str(indx+1)+' of '+str(len(cross_combinations))+' combinations ...')
  train_log = []
  test_log = []
  comb_log = {}
  # Hyperparameters
  embedding_dim = hyper_parameters['embedding_dim']
  n_way = n_test_way = hyper_parameters['n_way']
  n_shot = n_test_shot = hyper_parameters['n_shot']
  n_query = n_test_query = hyper_parameters['n_query']
  
  h_dim, z_dim = embedding_dim, embedding_dim

  print("hyper_parameters: ",hyper_parameters)
  # SetUp Model ------------------------------------------------------------------------------------------------------
  # ---------------------------------------------------------------------------------------------------------------------
  tf.reset_default_graph()
  x = tf.placeholder(tf.float32, [None, None, im_height, im_width, channels])
  q = tf.placeholder(tf.float32, [None, None, im_height, im_width, channels])
  x_shape = tf.shape(x)
  q_shape = tf.shape(q)
  num_classes, num_support = x_shape[0], x_shape[1]
  num_queries = q_shape[1]
  y = tf.placeholder(tf.int64, [None, None])
  y_one_hot = tf.one_hot(y, depth=num_classes)
  
  if convnet_activate:
    emb_x = conv_net(tf.reshape(x, [num_classes * num_support, im_height, im_width, channels]), embedding_dim)
  else:
    emb_x = encoder(tf.reshape(x, [num_classes * num_support, im_height, im_width, channels]), h_dim, z_dim)
  

  emb_dim = tf.shape(emb_x)[-1]
  emb_x = tf.reduce_mean(tf.reshape(emb_x, [num_classes, num_support, emb_dim]), axis=1)
  
  if convnet_activate:
    emb_q = conv_net(tf.reshape(q, [num_classes * num_queries, im_height, im_width, channels]),embedding_dim)
  else:
    emb_q = encoder(tf.reshape(q, [num_classes * num_queries, im_height, im_width, channels]), h_dim, z_dim, reuse=True)
  

  dists = euclidean_distance(emb_q, emb_x)
  log_p_y = tf.reshape(tf.nn.log_softmax(-dists), [num_classes, num_queries, -1])
  ce_loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_one_hot, log_p_y), axis=-1), [-1]))
  acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(log_p_y, axis=-1), y)))

  # Optimizer
  train_op = tf.train.AdamOptimizer().minimize(ce_loss)

  # PUT THIS IN YOUR GRAPH
  var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)#, scope='head')
  gradios_amigos = tf.train.AdamOptimizer().compute_gradients(ce_loss, var_list=var_list)
  mi_gradios_amigos = [g for g in gradios_amigos if g[0] is not None]

  # Session
  sess = tf.InteractiveSession()
  init_op = tf.global_variables_initializer()
  sess.run(init_op)


  # Training Model ------------------------------------------------------------------------------------------------------
  # ---------------------------------------------------------------------------------------------------------------------
  stopping_criteria = 0
  last_test_acc = -1
  avg_acc = 0.
  avg_ls = 0.
  acc_train = []
  lss_train = []
  gradient_list_train = []
  acc_test = []
  lss_test = []
  gradient_list_test = []

  for ep in range(n_epochs):
      print('------------------------------------------------------------')
      print('EPOCH No.',ep)
      L2_tmp = []
      for epi in range(n_episodes):
          epi_classes = np.random.permutation(n_classes)[:n_way]
          support = np.zeros([n_way, n_shot, im_height, im_width,3], dtype=np.float32)
          query = np.zeros([n_way, n_query, im_height, im_width,3], dtype=np.float32)
          for i, epi_cls in enumerate(epi_classes):
              selected = np.random.permutation(n_examples)[:n_shot + n_query]
              support[i] = train_dataset[epi_cls, selected[:n_shot]]
              query[i] = train_dataset[epi_cls, selected[n_shot:]]
          #support = np.expand_dims(support, axis=-1)
          #query = np.expand_dims(query, axis=-1)
          labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query)).astype(np.uint8)

          _, ls, ac , grads = sess.run([train_op, ce_loss, acc, mi_gradios_amigos], feed_dict={x: support, q: query, y:labels})
          avg_acc += ac
          avg_ls += ls
          # compute L2 of this minibatch and append to list         
          L_dos = np.linalg.norm(np.hstack([g[0].flatten() for g in grads]), ord=2)
          L2_tmp.append(L_dos)

          if (epi+1) % 50 == 0:
              print('[epoch {}/{}, episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(ep+1, n_epochs, epi+1, n_episodes, ls, ac))
      
      gradient_list_train.append(np.mean(L2_tmp))
      avg_acc /= (n_episodes)
      avg_ls /= (n_episodes)

      acc_train.append(avg_acc)
      lss_train.append(avg_ls)
   
      print('Average Train Accuracy: {:.5f}'.format(avg_acc))
      print('Average Train Loss: {:.5f}'.format(avg_ls))

      print('L2-norm of gradients in epochs {}:'.format(ep))
      print(np.mean(L2_tmp))

      train_log = (acc_train,lss_train,gradient_list_train)

      print('Testing....')
      avg_acc_test = 0.
      avg_ls_test = 0.
      L2_tmp_test = []
      if ep % 20 == 0 and ep > 10:#20-10
        for ep in range(n_test_epochs):
          
          for epi in range(n_test_episodes):
              epi_classes = np.random.permutation(n_test_classes)[:n_test_way]
              support = np.zeros([n_test_way, n_test_shot, im_height, im_width,3], dtype=np.float32)
              query = np.zeros([n_test_way, n_test_query, im_height, im_width,3], dtype=np.float32)
              for i, epi_cls in enumerate(epi_classes):
                  selected = np.random.permutation(n_examples_test)[:n_test_shot + n_test_query]
                  support[i] = test_dataset[epi_cls, selected[:n_test_shot]]
                  query[i] = test_dataset[epi_cls, selected[n_test_shot:]]
              #support = np.expand_dims(support, axis=-1)
              #query = np.expand_dims(query, axis=-1)
              labels = np.tile(np.arange(n_test_way)[:, np.newaxis], (1, n_test_query)).astype(np.uint8)
              ls, ac , grads= sess.run([ce_loss, acc, mi_gradios_amigos], feed_dict={x: support, q: query, y:labels})
              avg_acc_test += ac
              avg_ls_test += ls

              # compute L2 of this minibatch and append to list         
              L_dos = np.linalg.norm(np.hstack([g[0].flatten() for g in grads]), ord=2)
              L2_tmp_test.append(L_dos)

              if (epi+1) % 50 == 0:
                  print('[test episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(epi+1, n_test_episodes, ls, ac))

        gradient_list_test.append(np.mean(L2_tmp_test))
        avg_acc_test /= (n_test_episodes * n_test_epochs)
        avg_ls_test /= (n_test_episodes* n_test_epochs)

        acc_test.append(avg_acc_test)
        lss_test.append(avg_ls_test)
    
        print('Average Test Accuracy: {:.5f}'.format(avg_acc_test))
        print('Average Test Loss: {:.5f}'.format(avg_ls_test))

        print('L2-norm of gradients in Test epochs {}:'.format(ep))
        print(np.mean(L2_tmp_test))
        
        test_log = (acc_test,lss_test, gradient_list_test)

        comb_log['train'] = train_log
        comb_log['test'] = test_log
        comb_log['params'] = hyper_parameters
      
        log_performance = comb_log
              # Stopping Criteria
        if avg_acc_test <= (last_test_acc * 1.005):
            print('- - - Model Not Improved - - -')
            stopping_criteria += 1
        else:
            print('- - - Model Improved - - -')
            stopping_criteria = 0
            last_test_acc = avg_acc_test

        if stopping_criteria >= 3:
            print('- - - Finished Learning: Stopping Criteria - - -')
            break

pickle_out = open(filename,"wb")
pickle.dump(log_performance, pickle_out)
pickle_out.close()

