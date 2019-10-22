from __future__ import absolute_import, division, print_function, unicode_literals
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization


import tensorflow as tf
import numpy as np








def conv_netV2(x, embedding_size=100):
    conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 32], mean=0, stddev=0.1))
    conv1_filter2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 32], mean=0, stddev=0.1))

    conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 64], mean=0, stddev=0.1))
    conv2_filter2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], mean=0, stddev=0.1))
   

    # conv 1 block
    conv1 = tf.nn.conv2d(x, conv1_filter, strides=[1,1,1,1], padding='SAME')
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.conv2d(conv1, conv1_filter2, strides=[1,1,1,1], padding='VALID')
    conv1 = tf.nn.relu(conv1)
    conv1_pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    conv1_pool = tf.nn.dropout(conv1_pool, keep_prob=0.75)

    # conv 2 block
    conv2 = tf.nn.conv2d(conv1_pool, conv2_filter, strides=[1,1,1,1], padding='SAME')
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.conv2d(conv2, conv2_filter2, strides=[1,1,1,1], padding='VALID')
    conv2 = tf.nn.relu(conv2)
    conv2_pool = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    conv2_pool = tf.nn.dropout(conv2_pool, keep_prob=0.75)

    # dense block
    flat = tf.contrib.layers.flatten(conv2_pool)  
    full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=512, activation_fn=tf.nn.relu)
    full1 = tf.nn.dropout(full1, keep_prob=0.5) 
    full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=embedding_size, activation_fn=tf.nn.sigmoid)
    full2 = tf.layers.batch_normalization(full2)
    #full2 = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=10, activation_fn=None)
    return full2 # shape = (batch_size, embeddingsize) 

