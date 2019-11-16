# -*- coding: utf-8 -*-
# """Varkeys+Resnet.ipynb

# Automatically generated by Colaboratory.

# Original file is located at
#     https://colab.research.google.com/drive/1uXbDMOrv7s7ZRkCqzrsqbSWPbcBFKkWP
# """

# import keras
# import tensorflow as tf  

# import numpy as np
    
# from keras.datasets import cifar10
# #from tensorflow.keras.applications import ResNet50


# from keras.applications.resnet50 import ResNet50
# from keras.layers import GlobalAveragePooling2D, Dense
# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Model
# from keras.layers import Layer
# import matplotlib.pyplot as plt

# import sys

# # set default parameters
# KEY_SIZE = 100  # keysize (= embedding size)
# NUM_KEYS = 1000 # number of keys per class
# LR=0.0001       # learning rate
# BANDWIDTH = 10000 # bandwith parameter
# MEMORY = 1

# NUM_CLASSES = 10
# EPOCHS = 320


# KEY_SIZE = int(KEY_SIZE)
# NUM_KEYS = int(NUM_KEYS)
# LR = float(LR)
# BANDWIDTH = int(BANDWIDTH)
# MEMORY = int(MEMORY)

# #sys.exit(0)


# # load data
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()



# # Input image dimensions.
# input_shape = x_train.shape[1:]

# # Normalize data.
# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255

# # Convert class vectors to binary class matrices.
# y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
# y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)





# class Varkeys(Layer):

#     def __init__(self, keysize, dict_size, values, categories, **kwargs):
#         self.output_dim = keysize
#         self.initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=None)
#         self.values = values
#         self.categories = categories
#         self.keysize = keysize 
#         self.dict_size = dict_size
#         super(Varkeys, self).__init__(**kwargs)

#     def build(self, input_shape):
#         # Create a trainable weight variable for this layer.
#         self.keys = self.add_weight(name='keys', 
#                                       shape=(self.dict_size, self.keysize),
#                                       initializer=self.initializer,
#                                       trainable=True)
        
#         super(Varkeys, self).build(input_shape)  # Be sure to call this at the end


#     def call(self, x):
#         KV =  tf.matmul(tf.transpose(self.kernel(self.keys, x)), self.values)
#         KV_ = tf.diag(tf.reshape(tf.reciprocal( tf.matmul(KV,tf.ones((self.categories,1)))) , [-1]))

#         return tf.matmul(KV_, KV)

#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], self.categories)

    
#     def sq_distance(self, A, B):
#         print('im in distance function')
#         row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
#         row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

#         row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
#         row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

#         return row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B

#     def kernel (self, A,B):
#         print('im in kernel function!!')
#         d = self.sq_distance(A,B) / BANDWIDTH
#         o = tf.reciprocal(d+1e-4)
#         return o  

# def one_hot(length, i):
#     return [1 if idx==i else 0 for idx in range(length)]



# values = [one_hot(10, i) for i in range(NUM_CLASSES)] * NUM_KEYS
# n_keys= len(values)
# n_output = y_train.shape[1]
# V = tf.constant(values, dtype=tf.float32, shape = (n_keys, n_output))


# base_model = ResNet50(weights='imagenet',include_top=False, input_shape=input_shape)

# # freeze pretrained model
# for layer in base_model.layers[1:-2]:
#     layer.trainable=False

# x = base_model.output
# x = GlobalAveragePooling2D()(x)

# # use memory:
# if MEMORY == 1:
#     x = Dense(KEY_SIZE, activation='relu')(x)
#     predictions = Varkeys(KEY_SIZE, n_keys, V, NUM_CLASSES)(x)

# else:
#     x = Dense(512, activation='relu')(x)
#     # and a logistic layer -- 10 classes for CIFAR10
#     predictions = Dense(NUM_CLASSES, activation='softmax')(x)


# # this is the model we will train
# model = Model(inputs=base_model.input, outputs=predictions)




# model.compile(
#     #optimizer='Adam',
#     optimizer=keras.optimizers.Adam(lr=LR),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# # tensorboard
# tbCallBack = keras.callbacks.TensorBoard(log_dir='tb_logs/lr={}&memory={}'.format(LR, MEMORY), histogram_freq=0, write_graph=True, write_images=True)


# history = model.fit(x_train, y_train, validation_data=(x_test, y_test),  epochs=EPOCHS, verbose=2, callbacks = [tbCallBack])
# model.evaluate(x_test, y_test, verbose=1)




# plt.plot(history.history['acc'])#
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# if MEMORY == 1:
#     plt.savefig('acc_lr={}.png'.format(KEY_SIZE, NUM_KEYS, LR, BANDWIDTH) )
# else:
#     plt.savefig('acc_momory=0&lr={}.png'.format(LR, BANDWIDTH))
# plt.clf()

# plt.plot(history.history['loss'])#
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# if MEMORY == 1:
#     plt.savefig('loss_lr={}.png'.format(KEY_SIZE, NUM_KEYS, LR, BANDWIDTH) )
# else:
#     plt.savefig('loss_momory=0&lr={}.png'.format(LR, BANDWIDTH))   
# plt.clf()

# base_model = ResNet50(weights='imagenet',include_top=False, input_shape=input_shape)

# output = base_model(x_train)

"""Varkeys+ *tensorflow*"""

#http://zachmoshe.com/2017/11/11/use-keras-models-with-tf.html
#https://github.com/zachmoshe/zachmoshe.com/blob/master/content/use-keras-models-with-tf/using-keras-models-in-tf.ipynb
import keras
import tensorflow as tf  

import numpy as np
    
from keras.datasets import cifar10
#from tensorflow.keras.applications import ResNet50
NUM_CLASSES = 10

from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Layer
import matplotlib.pyplot as plt

import sys



# load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#https://gist.github.com/JVGD/2add7789ab83588a397bbae6ff614dbf
#https://github.com/frlim/data2040_final/blob/master/project_2/CNN_Final.ipynb

# Input image dimensions.
input_shape = x_train.shape[1:]
input_shape = (200,200,3)

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

import numpy as np
import tensorflow as tf
import tempfile
import math

DEFAULT_IMAGE_SHAPE = (None,200,200,3)

class Resnet():
    """
    A class that builds a TF graph with a pre-trained VGG19 model (on imagenet)
    Also takes care of preprocessing. Input should be a regular RGB image (0-255)
    """
    def __init__(self, image_shape=DEFAULT_IMAGE_SHAPE, input_tensor=None):
        self.image_shape = image_shape
        self._build_graph(input_tensor)

    def _build_graph(self, input_tensor):
        with tf.Session() as sess:
            with tf.variable_scope('Resnet'):
                with tf.name_scope('inputs'):
                    if input_tensor is None:
                        input_tensor = tf.placeholder(tf.float32, shape=self.image_shape, name='input_img')
                    #else:
                        #assert self.image_shape == input_tensor.shape
                    self.input_tensor = input_tensor


                with tf.variable_scope('model'):
                    self.resnet = ResNet50(weights='imagenet',include_top=False, input_shape=input_shape, input_tensor = self.input_tensor)

                self.outputs = { l.name: l.output for l in self.resnet.layers }

            self.resnet_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Resnet/model')
            
            with tempfile.NamedTemporaryFile() as f:
                self.tf_checkpoint_path = tf.train.Saver(self.resnet_weights).save(sess, f.name)

        self.model_weights_tensors = set(self.resnet_weights)

    def freeze(self):

      print("freezing...")
      for layer in self.resnet.layers:
        layer.trainable = False

    def unfreeze(self):

      print("unfreezing...")
      for layer in self.resnet.layers:
        layer.trainable = True

    def load_weights(self):
        sess = tf.get_default_session()
        tf.train.Saver(self.resnet_weights).restore(sess, self.tf_checkpoint_path)

    def __getitem__(self, key):
        return self.outputs[key]

import os


tf.reset_default_graph()



l = 50
learning_rate = 0.00002
batch_size = 64

trainable_weights_graph = tf.Graph()
epsilon = 1e-3
with trainable_weights_graph.as_default():

  x = tf.placeholder(tf.float32, (None,32,32,3), name='my_original_image')
  y = tf.placeholder(tf.float32, shape=(None, 10), name='output_y')

#   x2 = tf.image.resize(x, (200,200))
#   x2 = tf.compat.v1.image.resize(x, (200,200))

  resnet = Resnet(image_shape=(None,200,200,3), input_tensor=x2)
  W1 = tf.Variable(tf.truncated_normal([2048, 128], stddev=0.1))
  W2 = tf.Variable(tf.truncated_normal([128, 64], stddev=0.1))
  W3 = tf.Variable(tf.truncated_normal([64, 10], stddev=0.1))

  b1 = tf.Variable(tf.constant(0.1, shape=[128]))
  b2 = tf.Variable(tf.constant(0.1, shape=[64]))
  b3 = tf.Variable(tf.constant(0.1, shape=[10]))


  output1 = tf.identity(resnet['activation_49'], name='my_output')
  output1_avg = tf.nn.avg_pool(output1, ksize=[1,7,7,1], strides=[1,7,7,1], padding='VALID')
  output2 = tf.reshape(output1_avg, (-1,2048))

  
 


  #bn_output2 = tf.layers.batch_normalization(output2)
  #output2d = tf.nn.dropout(bn_output2, 0.9)

  output3 =  tf.linalg.matmul( output2,W1)+b1

  batch_mean2, batch_var2 = tf.nn.moments(output3,[0])
  scale2 = tf.Variable(tf.ones([128]))
  beta2 = tf.Variable(tf.zeros([128]))
  output3d = tf.nn.batch_normalization(output3,batch_mean2,batch_var2,beta2,scale2,epsilon)

  routput3 = tf.nn.relu(output3d)
  #d_routput3 = tf.nn.dropout(routput3, 0.9)
 # bn_routput3 = tf.layers.batch_normalization(d_routput3)

  output4 =  tf.linalg.matmul( routput3,W2)+b2
  routput4 = tf.nn.relu(output4)

  output = tf.linalg.matmul( routput4, W3) +b3

  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))  
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


  # Accuracy
  correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

resnet.resnet.summary()

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

acc_train_list =  []
acc_test_list = []
num_epochs = 100

with tf.Session(graph=trainable_weights_graph) as sess:

    sess.run(tf.global_variables_initializer())
    resnet.freeze()
    resnet.load_weights()
    

    print(np.sum(np.mean(resnet.resnet.get_layer('res5b_branch2a').get_weights())))

    for epoch in range(num_epochs):
      print("This is gonna work:")
      print("-----------------------------------------")


      minibatches = random_mini_batches(x_train, y_train, batch_size, 1)
      acc_temp = []
      
      
      for i, minibatch in enumerate(minibatches):
          batch_X, batch_Y = minibatch
          _,  o, cost_train, acc_train = sess.run([optimizer,output, cost, accuracy], 
                                  feed_dict={x: batch_X, 
                                              y: batch_Y})
          acc_temp.append(acc_train)


      acc_train_list.append(np.mean(acc_temp))

      minibatches = random_mini_batches(x_test, y_test, batch_size, 1)
      acc_temp = []
      for i, minibatch in enumerate(minibatches):
        batch_X, batch_Y = minibatch
        cost_test, acc_test = sess.run([cost, accuracy], 
                                  feed_dict={x: batch_X, 
                                              y: batch_Y})
        acc_temp.append(acc_test)
      acc_test_list.append(np.mean(acc_temp))

      print("Epoch ", epoch, " ...")
      #print("Cost train:", np.mean(acc_temp))
      print("Acc test:", acc_test_list[-1])
      #print("Cost test:", cost_train)
      print("Acc train:", acc_train_list[-1])

np.mean(acc_temp)

with tf.Session(graph=trainable_weights_graph) as sess:
  sess.run(tf.global_variables_initializer())
  x2_ = sess.run(output2,feed_dict={x: batch_X, y: batch_Y} )
  print(np.sum(np.mean(resnet.resnet.get_layer('res5b_branch2a').get_weights())))

resnet.resnet.summary()

plt.plot(acc_train_list)
plt.plot(acc_test_list)

plt.plot(acc_train_list)
plt.plot(acc_test_list)

plt.plot(acc_train_list)
plt.plot(acc_test_list)

with tf.Session(graph=trainable_weights_graph) as sess:
    resnet.load_weights()
    fd = { x: x_train[0][np.newaxis,:] }

    output_val = sess.run(output, fd)

with tf.Session(graph=trainable_weights_graph) as sess:
  print(resnet.resnet.get_layer('bn5c_branch2c').trainable)

output_val.shape

y_train[1]

#https://github.com/frlim/data2040_final/blob/master/project_2/CNN_Final.ipynb

#https://gist.github.com/JVGD/2add7789ab83588a397bbae6ff614dbf

#https://github.com/PrzemekPobrotyn/CIFAR-10-transfer-learning/blob/master/report.ipynb

#http://zachmoshe.com/2017/11/11/use-keras-models-with-tf.html
#https://github.com/zachmoshe/zachmoshe.com/blob/master/content/use-keras-models-with-tf/using-keras-models-in-tf.ipynb
#https://github.com/zachmoshe/zachmoshe.com/blob/master/content/use-keras-models-with-tf/using-keras-models-in-tf.ipynb

#Fine tune pretrained models: https://medium.com/datadriveninvestor/deep-learning-using-transfer-learning-python-code-for-resnet50-8acdfb3a2d38
#https://github.com/frlim/data2040_final/blob/master/project_2/CNN_Final.ipynb