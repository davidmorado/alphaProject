from __future__ import absolute_import, division, print_function, unicode_literals
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
import numpy as np

def Stage1(x, embedding_size=100):
    with tf.variable_scope('Stage1', reuse=tf.AUTO_REUSE):
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

    return full2 # shape = (batch_size, embeddingsize) 

def Stage2(h, embedding_size=100, target_size=10, reuse=False):
    with tf.variable_scope('Stage2', reuse=reuse):
        full1 = tf.contrib.layers.fully_connected(inputs=h, num_outputs=int(embedding_size/2), activation_fn=tf.nn.relu)
        bn1 = tf.contrib.layers.batch_norm(full1)
        full2 = tf.contrib.layers.fully_connected(inputs=bn1, num_outputs=int(embedding_size/4), activation_fn=tf.nn.relu)
        bn2 = tf.contrib.layers.batch_norm(full2)
        logits = tf.contrib.layers.fully_connected(inputs=bn2, num_outputs=target_size, activation_fn=None) 
    return logits









































# def Stage2(h, embedding_size=100, target_size=10, reuse=None):
#     with simple_variable_scope('Stage2', reuse=reuse):
#         full1 = tf.contrib.layers.fully_connected(inputs=h, num_outputs=int(embedding_size/2), activation_fn=tf.nn.relu)
#         bn1 = tf.contrib.layers.batch_norm(full1)
#         full2 = tf.contrib.layers.fully_connected(inputs=bn1, num_outputs=int(embedding_size/4), activation_fn=tf.nn.relu)
#         bn2 = tf.contrib.layers.batch_norm(full2)
#         logits = tf.contrib.layers.fully_connected(inputs=bn2, num_outputs=target_size, activation_fn=None) 
#     return logits


# def Stage2(h, embedding_size=100, target_size=10, reuse=None):
#     with tf.variable_scope('Stage2', reuse=tf.AUTO_REUSE) as scope:
#         full1 = tf.contrib.layers.fully_connected(inputs=h, num_outputs=int(embedding_size/2), activation_fn=tf.nn.relu, scope=scope, reuse=reuse)
#         bn1 = tf.contrib.layers.batch_norm(full1, scope=scope, reuse=reuse)
#         # full1 = tf.layers.batch_normalization(full1)
#         full2 = tf.contrib.layers.fully_connected(inputs=bn1, num_outputs=int(embedding_size/4), activation_fn=tf.nn.relu, scope=scope, reuse=reuse)
#         bn2 = tf.contrib.layers.batch_norm(full2, scope=scope, reuse=reuse)
#         # full2 = tf.layers.batch_normalization(full2)
#         logits = tf.contrib.layers.fully_connected(inputs=bn2, num_outputs=target_size, activation_fn=None, scope=scope, reuse=reuse) 
#     return logits


# class SecondStage():

#     def __init__(self, embedding_size=100, target_size=10):
        
#         self.embedding_size = embedding_size
#         self.target_size = target_size
#         self.session = None
                

#         with tf.variable_scope('SECOND_STAGE', reuse=tf.AUTO_REUSE):
#             # self.h = tf.placeholder(tf.float32, shape=(None, self.embedding_size), name='h')
#             self.full1 = lambda h:tf.contrib.layers.fully_connected(inputs=h, num_outputs=int(self.embedding_size/2), activation_fn=tf.nn.relu)
#             self.logits = lambda full1: tf.contrib.layers.fully_connected(inputs=full1, num_outputs=target_size, activation_fn=None)
        
    
#     def __call__(self, h):
#         if self.session is None:
#             raise('No session assigned to instance of SecondStage')
        
#         #return self.session.run(self.logits, feed_dict={self.h : h})
#         with tf.variable_scope('SECOND_STAGE', reuse=tf.AUTO_REUSE):
#             return self.logits(self.full1(h))
        

#     def __init_session(self):
#         if self.session is None:
#             raise('No session assigned to instance of SecondStage')
#         self.session.run(tf.variables_initializer(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SECOND_STAGE')))


#     def set_session(self, sess):
#         self.session = sess 
#         #self.__init_session()