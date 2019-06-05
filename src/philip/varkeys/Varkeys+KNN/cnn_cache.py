# https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html
# https://www.tensorflow.org/guide/graphs



import keras
import tensorflow as tf  
from keras import backend as K

import numpy as np
    
from keras.datasets import cifar10
#from tensorflow.keras.applications import ResNet50


from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Layer
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
import keras
from keras.layers.normalization import BatchNormalization

import sys

# set default parameters
KEY_SIZE = 100  # keysize (= embedding size)
NUM_KEYS = 100 # number of keys per class
LR=0.0001       # learning rate
BANDWIDTH = 10000 # bandwith parameter
MEMORY = 1

NUM_CLASSES = 10
EPOCHS = 10
batch_size = 32


# read arguments
if len(sys.argv) == 6:
    KEY_SIZE, NUM_KEYS, LR, BANDWIDTH, MEMORY = sys.argv[1:]
    print('KEY_SIZE: ', KEY_SIZE)
    print('NUM_KEYS: ', NUM_KEYS)
    print('LR: ', LR)
    print('BANDWIDTH: ', BANDWIDTH)
    print('MEMORY: ', MEMORY)
    print('NUM_CLASSES: ', NUM_CLASSES)
    print('EPOCHS: ', EPOCHS)

# read arguments when using no memory
elif len(sys.argv) == 3:
    LR, MEMORY = sys.argv[1:]
    print('LR: ', LR)
    print('MEMORY: ', MEMORY)
    print('EPOCHS: ', EPOCHS)

else:
    # no arguments, print defualt ones
    print('KEY_SIZE: ', KEY_SIZE)
    print('NUM_KEYS: ', NUM_KEYS)
    print('LR: ', LR)
    print('BANDWIDTH: ', BANDWIDTH)
    print('MEMORY: ', MEMORY)
    print('NUM_CLASSES: ', NUM_CLASSES)
    print('EPOCHS: ', EPOCHS)

KEY_SIZE = int(KEY_SIZE)
NUM_KEYS = int(NUM_KEYS)
LR = float(LR)
BANDWIDTH = int(BANDWIDTH)
MEMORY = int(MEMORY)

#sys.exit(0)


# create folders
import os

folders = ['data', 'plots', 'logs', 'err_logs', 'tb_logs']
for f in folders:
    try:
        os.makedirs(f)
    except OSError:
        pass




# load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()



# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)





class Varkeys(Layer):

    def __init__(self, keysize, dict_size, values, categories, **kwargs):
        self.output_dim = keysize
        self.initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=None)
        self.values = values
        self.categories = categories
        self.keysize = keysize 
        self.dict_size = dict_size
        self.k = 8
        self.knn = KNN(self.k)
        super(Varkeys, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.keys = self.add_weight(name='keys', 
                                      shape=(self.dict_size, self.keysize),
                                      initializer=self.initializer,
                                      trainable=True)

        # build KNN  
        # placeholders for KNN
        self.pl_x = tf.placeholder(tf.float32, [None, self.keysize], name='input')  # (batch_size x embedding_size)
        self.pl_keys = tf.placeholder(tf.float32, [self.dict_size, self.keysize], name='keys') # (n_keys x embedding_size)
        self.pl_values = tf.placeholder(tf.int32, [self.dict_size, self.categories], name='values') # (n_keys x n_classes) 

        
        super(Varkeys, self).build(input_shape)  # Be sure to call this at the end


    def call(self, x):
        #init_op = tf.global_variables_initializer()
        #output = self.knn.build(self.pl_x, self.pl_keys, self.pl_values)
        #feed_dict = {self.pl_x : x, self.pl_keys : self.keys, self.pl_values : self.values}
        output = self.knn.build(x, self.keys, self.values)
        # with tf.Session() as sess:
        #     sess.run(init_op)
        #     return sess.run([output], feed_dict=feed_dict)
        sess = K.get_session()
        with sess.as_default():
            return output

        #return sess.run([output])
        





    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.categories)


class KNN:

    def __init__(self, k):
        self.k = k # hyperparameter k for KNN


    def build(self, x, keys, values):
        # x:  (batch_size x embedding_size)
        # keys (n_keys x embedding_size)
        # values (n_keys x n_classes)
        with tf.variable_scope('KNN'):
            expanded_x = tf.expand_dims(x, axis=1) # shape: (batch_size, 1, embedding size)
            expanded_keys = tf.expand_dims(keys, axis=0) # shape: (1, dict_size, embedding_size)
            tiled_expanded_x = tf.tile(expanded_x, [1, tf.shape(keys)[0], 1]) # shape: (batch_size, dict_size, embedding size)

            # compute distances
            diff = tf.square(expanded_keys - tiled_expanded_x)
            distances = tf.reduce_sum(diff,axis=2)  # shape: (batch_size, dict_size)

            # get nearest neighbors
            _, indices = tf.nn.top_k(-distances, k=self.k)  
            #hit_keys = tf.nn.embedding_lookup(self.keys, indices) # (batch size, k, embedding_size)
            hit_values = tf.nn.embedding_lookup(values, indices) # shape: (batch-size, k, n_classes) 
            print(hit_values.shape)

        with tf.control_dependencies([hit_values]):
            # majority class vote
            sum_up_predictions = tf.reduce_sum(hit_values, axis=1) # shape : (batch size, n_classes)
            #out  = tf.argmax(sum_up_predictions, axis=1)
            end_node = sum_up_predictions / tf.constant(self.k, dtype=tf.float32)
            return end_node

     







def one_hot(length, i):
    return [1 if idx==i else 0 for idx in range(length)]



values = [one_hot(10, i) for i in range(NUM_CLASSES)] * NUM_KEYS
n_keys= len(values)
n_output = y_train.shape[1]
V = tf.constant(values, dtype=tf.float32, shape = (n_keys, n_output))







def CNN_keys(layers=[32, 64, 512], embedding_dim = 20, num_classes=10, n_keys= n_keys, V=[]):
        
    model = Sequential()

    model.add(Conv2D(layers[0], (3, 3), padding='same',
                    input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(layers[0], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(layers[1], (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(layers[1], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(layers[2]))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(embedding_dim))
    model.add(Activation('sigmoid'))
    model.add(BatchNormalization())
    model.add(Varkeys(embedding_dim, n_keys, V, num_classes))  # arguments: keysize, dict_size, values, categories
    return model


def CNN(layers=[32, 64, 512], embedding_dim = 20, num_classes=10):

    model = Sequential()

    model.add(Conv2D(layers[0], (3, 3), padding='same',
                    input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(layers[0], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(layers[1], (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(layers[1], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(layers[2]))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(embedding_dim))
    model.add(Activation('sigmoid'))
    model.add(BatchNormalization())
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))  

    return model


# use memory:
if MEMORY == 1:
    model = CNN_keys(embedding_dim = KEY_SIZE, num_classes=NUM_CLASSES, n_keys= n_keys, V=V)

else:
    model = CNN(embedding_dim = KEY_SIZE, num_classes=NUM_CLASSES)


model.compile(loss=keras.losses.categorical_crossentropy,
            # optimizer=keras.optimizers.SGD(lr=0.1),
            optimizer = keras.optimizers.rmsprop(lr=LR, decay=1e-6),
            metrics=['accuracy'])
# model.compile(
#     #optimizer='Adam',
#     optimizer=keras.optimizers.Adam(lr=LR),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# tensorboard
#tbCallBack = keras.callbacks.TensorBoard(log_dir='tb_logs/lr={}&memory={}'.format(LR, MEMORY), histogram_freq=0, write_graph=True, write_images=True)


#history = model.fit(x_train, y_train, validation_data=(x_test, y_test),  epochs=EPOCHS, verbose=2, callbacks = [tbCallBack], batch_size=batch_size)
history = model.fit(x_train, y_train, validation_data=(x_test, y_test),  epochs=EPOCHS, verbose=2, batch_size=batch_size)

model.evaluate(x_test, y_test, verbose=1)




plt.plot(history.history['acc'])#
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
if MEMORY == 1:
    plt.savefig('plots/acc_lr={}.png'.format(LR) )
else:
    plt.savefig('plots/acc_momory=0&lr={}.png'.format(LR))
plt.clf()

plt.plot(history.history['loss'])#
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
if MEMORY == 1:
    plt.savefig('plots/loss_lr={}.png'.format(LR) )
else:
    plt.savefig('plots/loss_momory=0&lr={}.png'.format(LR))   
plt.clf()

if MEMORY == 1:
    np.save('data/acc_lr={}'.format(LR), history.history['acc'])
    np.save('data/val_acc_lr={}'.format(LR), history.history['val_acc'])

    np.save('data/loss_lr={}'.format(LR), history.history['loss'])
    np.save('data/val_loss_lr={}'.format(LR), history.history['val_loss'])
else:
    np.save('data/noCache_acc_lr={}'.format(LR), history.history['acc'])
    np.save('data/noCache_val_acc_lr={}'.format(LR), history.history['val_acc'])

    np.save('data/noCache_loss_lr={}'.format(LR), history.history['loss'])
    np.save('data/noCache_val_loss_lr={}'.format(LR), history.history['val_loss'])

  

















































    # def call(self, x):

    #     ''' 
    #     x: batch_size x embedding size
    #     '''
    #     with tf.variable_scope('KNN'):
    #         expanded_x = tf.expand_dims(x, axis=1) # shape: (batch_size, 1, embedding size)
    #         expanded_keys = tf.expand_dims(self.keys, axis=0) # shape: (1, dict_size, embedding_size)
    #         tiled_expanded_x = tf.tile(expanded_x, [1, tf.shape(self.keys)[0], 1]) # shape: (batch_size, dict_size, embedding size)

    #         # compute distances
    #         diff = tf.square(expanded_keys - tiled_expanded_x)
    #         distances = tf.reduce_sum(diff,axis=2)  # shape: (batch_size, dict_size)

    #         # get nearest neighbors
    #         _, indices = tf.nn.top_k(-distances, k=self.k)  
    #         #hit_keys = tf.nn.embedding_lookup(self.keys, indices) # (batch size, k, embedding_size)
    #         hit_values = tf.nn.embedding_lookup(self.values, indices) # shape: (batch-size, k, n_classes) 
    #         print(hit_values.shape)

    #     with tf.control_dependencies([hit_values]):
    #         # majority class vote
    #         sum_up_predictions = tf.reduce_sum(hit_values, axis=1) # shape : (batch size, n_classes)
    #         #out  = tf.argmax(sum_up_predictions, axis=1)
    #         end_node = sum_up_predictions / tf.constant(self.k, dtype=tf.float32)
    #         return end_node


