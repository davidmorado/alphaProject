#http://zachmoshe.com/2017/11/11/use-keras-models-with-tf.html
#https://github.com/zachmoshe/zachmoshe.com/blob/master/content/use-keras-models-with-tf/using-keras-models-in-tf.ipynb
import keras
import tensorflow as tf  
import tempfile
import math
import numpy as np
    
from keras.datasets import cifar10
#from tensorflow.keras.applications import ResNet50


from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense, Layer, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras import layers
import matplotlib.pyplot as plt

import sys

#from data_loader import get_dataset, percentage_splitter

cifar = 10
tp = 0.1
embedding_dim = 100


# # get training data
# x_train, x_val, x_test, y_train, y_val, y_test = get_dataset('cifar{}'.format(cifar), normalize=True, ratio=0.2)
# # subsample training data
# x_train, y_train = percentage_splitter(x_train, x_val, y_train, y_val, merging=True, random_selection=False, ratio=tp) 
# num_categories = y_train.shape[1]
# N,h,w,c = x_train.shape
# input_shape=h,w,c




# load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#https://gist.github.com/JVGD/2add7789ab83588a397bbae6ff614dbf
#https://github.com/frlim/data2040_final/blob/master/project_2/CNN_Final.ipynb

# Input image dimensions.
input_shape = x_train.shape[1:]
input_shape = (32,32,3)

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

NUM_CLASSES = 10
# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)


DEFAULT_IMAGE_SHAPE = (None,256,256,3)

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
  x2 = tf.keras.layers.UpSampling2D((2,2))(x)
  x3 = tf.keras.layers.UpSampling2D((2,2))(x2)
  x4 = tf.keras.layers.UpSampling2D((2,2))(x3)
  # x4 = Reshape((200,200,3), input_shape=(32,32,3))(x)

  #x2 = tf.image.resize(x, (200,200))

  resnet = Resnet(image_shape=(None,256,256,3), input_tensor=x4)
  W1 = tf.Variable(tf.truncated_normal([2048, 128], stddev=0.1))
  W2 = tf.Variable(tf.truncated_normal([128, 64], stddev=0.1))
  W3 = tf.Variable(tf.truncated_normal([64, 10], stddev=0.1))

  b1 = tf.Variable(tf.constant(0.1, shape=[128]))
  b2 = tf.Variable(tf.constant(0.1, shape=[64]))
  b3 = tf.Variable(tf.constant(0.1, shape=[10]))


  output1 = tf.identity(resnet['activation_49'], name='my_output')
  output1_avg = tf.nn.avg_pool(output1, ksize=[1,7,7,1], strides=[1,7,7,1], padding='VALID')
  output2 = tf.reshape(output1_avg, (-1,2048))

  
#   output = GlobalAveragePooling2D()(output)
#   output = Dense(2048, activation='relu')(output)
#   output = Dense(128, activation='relu')(output)
#   output = Dense(embedding_dim, activation='relu')(output)
#   output = Dense(num_categories, activation='softmax')(output)
#   cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))  
#   optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)



  #bn_output2 = tf.layers.batch_normalization(output2)
  #output2d = tf.nn.dropout(bn_output2, 0.9)

  output3 =  tf.linalg.matmul(output2,W1)+b1

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
    #resnet.freeze()
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