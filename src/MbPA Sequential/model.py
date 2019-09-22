from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np



def conv_net(x, embedding_size=50):
    conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], mean=0, stddev=0.08))
    conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.08))
    conv3_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 128, 256], mean=0, stddev=0.08))
    conv4_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 256, 512], mean=0, stddev=0.08))

    # 1, 2
    conv1 = tf.nn.conv2d(x, conv1_filter, strides=[1,1,1,1], padding='SAME')
    conv1 = tf.nn.relu(conv1)
    conv1_pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv1_bn = tf.layers.batch_normalization(conv1_pool)

    # 3, 4
    conv2 = tf.nn.conv2d(conv1_bn, conv2_filter, strides=[1,1,1,1], padding='SAME')
    conv2 = tf.nn.relu(conv2)
    conv2_pool = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    
    conv2_bn = tf.layers.batch_normalization(conv2_pool)
  
    # 5, 6
    conv3 = tf.nn.conv2d(conv2_bn, conv3_filter, strides=[1,1,1,1], padding='SAME')
    conv3 = tf.nn.relu(conv3)
    conv3_pool = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')  
    conv3_bn = tf.layers.batch_normalization(conv3_pool)
    
    # 7, 8
    conv4 = tf.nn.conv2d(conv3_bn, conv4_filter, strides=[1,1,1,1], padding='SAME')
    conv4 = tf.nn.relu(conv4)
    conv4_pool = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv4_bn = tf.layers.batch_normalization(conv4_pool)
    
    # 9
    flat = tf.contrib.layers.flatten(conv4_bn)  

    # 10
    full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=128, activation_fn=tf.nn.relu)
    full1 = tf.layers.batch_normalization(full1)
    
    # 11
    full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=embedding_size, activation_fn=tf.nn.relu)
    full2 = tf.layers.batch_normalization(full2)
          
    
    # 14
    #out = tf.contrib.layers.fully_connected(inputs=full3, num_outputs=10, activation_fn=None)
    return full2 # shape = (batch_size, embeddingsize) 



def secondStage(h, embedding_size=50, target_size=10):
    with tf.variable_scope('SECOND_STAGE', reuse=tf.AUTO_REUSE):
        #full1 = tf.contrib.layers.fully_connected(inputs=h, num_outputs=int(embedding_size/2), activation_fn=tf.nn.relu, scope='SECOND_STAGE/layer1')
        full1 = tf.contrib.layers.fully_connected(inputs=h, num_outputs=int(embedding_size/2), activation_fn=tf.nn.relu)
        full1 = tf.layers.batch_normalization(full1)
        #full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=int(embedding_size/4), activation_fn=tf.nn.relu, scope='SECOND_STAGE/layer2')
        full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=int(embedding_size/4), activation_fn=tf.nn.relu)
        full2 = tf.layers.batch_normalization(full2)
        #logits = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=target_size, activation_fn=None, scope='SECOND_STAGE/layer3')
        logits = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=target_size, activation_fn=None)
    return logits


# tmp_weights = full1.weights 
# full1.weights = tmp_weights + adaptation
# # do prediction
# full1.weights = tmp_weights


