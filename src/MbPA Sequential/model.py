from __future__ import absolute_import, division, print_function, unicode_literals
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization


import tensorflow as tf
import numpy as np



def conv_netV2_orig(x,  embedding_size=100):
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










def conv_netV2(x, training_flag, embedding_size=100):
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
    conv1_pool = tf.layers.dropout(conv1_pool, rate=0.25, training=training_flag)

    # conv 2 block
    conv2 = tf.nn.conv2d(conv1_pool, conv2_filter, strides=[1,1,1,1], padding='SAME')
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.conv2d(conv2, conv2_filter2, strides=[1,1,1,1], padding='VALID')
    conv2 = tf.nn.relu(conv2)
    conv2_pool = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    conv2_pool = tf.layers.dropout(conv2_pool, rate=0.25, training=training_flag)

    # dense block
    flat = tf.contrib.layers.flatten(conv2_pool)  
    full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=512, activation_fn=tf.nn.relu)
    full1 = tf.layers.dropout(full1, rate=0.5, training=training_flag) 
    full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=embedding_size, activation_fn=tf.nn.sigmoid)
    full2 = tf.layers.batch_normalization(full2)
    #full2 = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=10, activation_fn=None)
    return full2 # shape = (batch_size, embeddingsize) 



def secondStage(h, embedding_size=50, target_size=10):
    with tf.variable_scope('SECOND_STAGE', reuse=tf.AUTO_REUSE):
        #full1 = tf.contrib.layers.fully_connected(inputs=h, num_outputs=int(embedding_size/2), activation_fn=tf.nn.relu, scope='SECOND_STAGE/layer1')
        full1 = tf.contrib.layers.fully_connected(inputs=h, num_outputs=int(embedding_size/2), activation_fn=tf.nn.relu)
        #full1 = tf.layers.batch_normalization(full1, name='SS_BatchNorm1')
        #full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=int(embedding_size/4), activation_fn=tf.nn.relu, scope='SECOND_STAGE/layer2')
        #full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=int(embedding_size/4), activation_fn=tf.nn.relu)
        #full2 = tf.layers.batch_normalization(full2, name='SS_BatchNorm2')
        #logits = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=target_size, activation_fn=None, scope='SECOND_STAGE/layer3')
        logits = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=target_size, activation_fn=None)
    return logits









class Node():
    def __init__(self, cond, embedding_size, inbound_nodes=[] ):
        # serves as connection between first stage and second stage
        # solves the problem where secondStage is either connected the output node of first stage OR takes samples from the dictionary
        self.inbound_nodes = inbound_nodes
        self.embedding_size = embedding_size
        # self.outbound_node = tf.zeros([self.bs_placeholder, self.embedding_size], dtype=tf.float32)
        self.outbound_node = tf.zeros_like(self.inbound_nodes[0])
        self.cond = cond # if cond == 0: use memory keys, if cond == 1: use encoder
        
        
        if len(self.inbound_nodes) > 2:
            raise 'this Node can only handle 2 inbound nodes'



    def __call__(self):
        # ref: https://stackoverflow.com/questions/37063952/confused-by-the-behavior-of-tf-cond
        
        true_fn = lambda: self.inbound_nodes[0]
        false_fn = lambda: self.inbound_nodes[1]
        pred =tf.equal(self.cond, tf.constant(0))
        out = tf.cond(pred, true_fn, false_fn)
        return out








class SecondStage():

    def __init__(self, encoder, embedding_size=100, target_size=10):
        
        self.embedding_size = embedding_size
        self.target_size = target_size
        self.session = None  
        self.cond = tf.placeholder(tf.int32, shape=[], name="condition") 
                 

        with tf.variable_scope('SECOND_STAGE', reuse=tf.AUTO_REUSE):
            
            self.node_placeholder = tf.placeholder(tf.float32, shape=(None, self.embedding_size), name='node_placeholder')
            inbound_nodes = [self.node_placeholder, encoder]
            self.input = Node(cond = self.cond, inbound_nodes=inbound_nodes, embedding_size=self.embedding_size)()
            self.full1 = tf.contrib.layers.fully_connected(inputs=self.input, num_outputs=int(self.embedding_size/2), activation_fn=tf.nn.relu)
            self.logits = tf.contrib.layers.fully_connected(inputs=self.full1, num_outputs=target_size, activation_fn=None)
        
    
    def __call__(self):
        if self.session is None:
            raise('No session assigned to instance of SecondStage during __call__')
        
        with tf.variable_scope('SECOND_STAGE'):
            return self.logits
        

    def __init_session(self):
        if self.session is None:
            raise('Session is None')
        self.session.run(tf.variables_initializer(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SECOND_STAGE')))


    def set_session(self, sess):
        self.session = sess 
      
    def get_h(self):
        return self.node_placeholder

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SECOND_STAGE')





















































class SecondStage_old():

    def __init__(self, embedding_size=100, target_size=10):
        
        self.embedding_size = embedding_size
        self.target_size = target_size
        self.session = None
                

        with tf.variable_scope('SECOND_STAGE', reuse=tf.AUTO_REUSE):
            # self.h = tf.placeholder(tf.float32, shape=(None, self.embedding_size), name='h')
            self.full1 = lambda h:tf.contrib.layers.fully_connected(inputs=h, num_outputs=int(self.embedding_size/2), activation_fn=tf.nn.relu)
            self.logits = lambda full1: tf.contrib.layers.fully_connected(inputs=full1, num_outputs=target_size, activation_fn=None)
        
    
    def __call__(self, h):
        if self.session is None:
            raise('No session assigned to instance of SecondStage during __call__')
        
        #return self.session.run(self.logits, feed_dict={self.h : h})
        with tf.variable_scope('SECOND_STAGE', reuse=tf.AUTO_REUSE):
            return self.logits(self.full1(h))
        

    def __init_session(self):
        if self.session is None:
            raise('No session assigned to instance of SecondStage')
        self.session.run(tf.variables_initializer(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SECOND_STAGE')))


    def set_session(self, sess):
        self.session = sess 
        #self.__init_session()

    def get_variables(self):
        
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SECOND_STAGE')














# tmp_weights = full1.weights 
# full1.weights = tmp_weights + adaptation
# # do prediction
# full1.weights = tmp_weights


