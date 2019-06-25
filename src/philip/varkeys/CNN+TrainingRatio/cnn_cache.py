




import keras
import tensorflow as tf  

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
KEY_SIZE = 20  # keysize (= embedding size)
NUM_KEYS = 100 # number of keys per class
LR=0.0001       # learning rate
BANDWIDTH = 10000 # bandwith parameter
MEMORY = 1

NUM_CLASSES = 10
EPOCHS = 100
batch_size = 64


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
        super(Varkeys, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.keys = self.add_weight(name='keys', 
                                      shape=(self.dict_size, self.keysize),
                                      initializer=self.initializer,
                                      trainable=True)
        
        super(Varkeys, self).build(input_shape)  # Be sure to call this at the end


    def call(self, x):
        print(x.shape)
        print(tf.transpose(self.kernel(self.keys, x)).shape)
        print(self.values.shape)
        KV =  tf.matmul(tf.transpose(self.kernel(self.keys, x)), self.values)
        KV_ = tf.diag(tf.reshape(tf.reciprocal( tf.matmul(KV,tf.ones((self.categories,1)))) , [-1]))

        return tf.matmul(KV_, KV)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.categories)

    
    def sq_distance(self, A, B):
        print('im in distance function')
        row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
        row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

        row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
        row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

        return row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B

    def kernel (self, A,B):
        print('im in kernel function!!')
        d = self.sq_distance(A,B) / BANDWIDTH
        o = tf.reciprocal(d+1e-4)
        return o  

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
model1 = CNN_keys(embedding_dim = KEY_SIZE, num_classes=NUM_CLASSES, n_keys= n_keys, V=V)

# model without memory
model2 = CNN(embedding_dim = KEY_SIZE, num_classes=NUM_CLASSES)










def fit_evaluate( model, x_train, y_train, x_test,  y_test, batch_size, epochs, lr, use_memory, training_ratio):
    # tensorboard
    tbCallBack = keras.callbacks.TensorBoard(log_dir='tb_logs/lr={}&memory={}&TR={}'.format(lr, use_memory, training_ratio), histogram_freq=0, write_graph=True, write_images=True)

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer = keras.optimizers.rmsprop(lr=lr, decay=1e-6),
                metrics=['accuracy'])

    model.fit(x_train, y_train, 
                validation_data=(x_test, y_test),  
                epochs=epochs, 
                verbose=2, 
                callbacks = [tbCallBack], 
                batch_size=batch_size)

    scores_train = model.evaluate(x_train, y_train, verbose=0)
    scores_test = model.evaluate(x_test, y_test, verbose=0)
    print("Train \n%s: %.2f%%" % (model.metrics_names[1], scores_train[1]*100))
    print("Val \n%s: %.2f%%" % (model.metrics_names[1], scores_test[1]*100))

    return scores_train, scores_test


def sample(x, y, tr, n_classes):

    sample_x = []
    sample_y = []
    
    for category in range(n_classes):
        idx_category = [idx for idx in range(y.shape[0]) if  y[idx, category] == 1]
        x_tmp = x[idx_category]
        y_tmp = y[idx_category]
        n = int(x_tmp.shape[0] * tr)

        sample_x.append(x_tmp[:n])
        sample_y.append(y_tmp[:n])

    sample_x = np.concatenate(sample_x, axis=0)
    sample_y = np.concatenate(sample_y, axis=0)
    return (sample_x, sample_y)



    
training_scores_memory = [] # list of lists: [ [loss, accuracy], ...]
training_scores_no_memory = []

test_scores_memory = []
test_scores_no_memory = []

training_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
training_ratios = [0.1, 0.2, 0.3]
EPOCHS = 10

# training ratios:
for tr in training_ratios:

    x_train_sample, y_train_sample = sample(x_train, y_train, tr, NUM_CLASSES)
    print(x_train_sample.shape, y_train_sample.shape)

    # with memory
    score_train_memory, score_test_memory = fit_evaluate( model1, x_train_sample, y_train_sample, x_test,  y_test, batch_size=batch_size, epochs=EPOCHS, lr=LR, use_memory=1, training_ratio=tr)
    
    # without memory
    score_train_no_memory, score_test_no_memory = fit_evaluate( model2, x_train_sample, y_train_sample, x_test,  y_test, batch_size=batch_size, epochs=EPOCHS, lr=LR, use_memory=0, training_ratio=tr)


    training_scores_memory.append(score_train_memory)
    test_scores_memory.append(score_test_memory)

    training_scores_no_memory.append(score_train_no_memory)
    test_scores_no_memory.append(score_test_no_memory)


import pickle

with open("data/test_scores_memory", 'wb') as f:
    pickle.dump(test_scores_memory, f)

with open("data/test_scores_no_memory", 'wb') as f:
    pickle.dump(test_scores_no_memory, f)

# load with:
# mynewlist = pickle.load(f)

# plot model performance with memory
test_loss_memory, test_acc_memory = zip(*test_scores_memory)
test_loss_no_memory, test_acc_no_memory = zip(*test_scores_no_memory)

plt.plot(training_ratios, test_acc_memory, label='memory')#
plt.plot(training_ratios, test_acc_no_memory, label='no memory')
plt.title('Comparison on different training ratios')
plt.ylabel('test accuracy')
plt.xlabel('training ratio')
plt.xticks(training_ratios)
plt.legend()
plt.savefig('plots/acc_lr={}.png'.format(LR) )
plt.clf()

plt.plot(training_ratios, test_loss_memory, label='memory')#
plt.plot(training_ratios, test_loss_no_memory, label='no memory')
plt.title('Comparison on different training ratios')
plt.ylabel('test loss')
plt.xlabel('training ratio')
plt.xticks(training_ratios)
plt.legend()
plt.savefig('plots/loss_lr={}.png'.format(LR) )
plt.clf()
  