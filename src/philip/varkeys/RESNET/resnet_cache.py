




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

import sys

# set default parameters
KEY_SIZE = 100  # keysize (= embedding size)
NUM_KEYS = 1000 # number of keys per class
LR=0.0001       # learning rate
BANDWIDTH = 10000 # bandwith parameter
MEMORY = 1

NUM_CLASSES = 10
EPOCHS = 320


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


base_model = ResNet50(weights='imagenet',include_top=False, input_shape=input_shape)

# freeze pretrained model
for layer in base_model.layers[1:-2]:
    layer.trainable=False

x = base_model.output
x = GlobalAveragePooling2D()(x)

# use memory:
if MEMORY == 1:
    x = Dense(KEY_SIZE, activation='relu')(x)
    predictions = Varkeys(KEY_SIZE, n_keys, V, NUM_CLASSES)(x)

else:
    x = Dense(512, activation='relu')(x)
    # and a logistic layer -- 10 classes for CIFAR10
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)


# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)




model.compile(
    #optimizer='Adam',
    optimizer=keras.optimizers.Adam(lr=LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# tensorboard
tbCallBack = keras.callbacks.TensorBoard(log_dir='tb_logs/lr={}&memory={}'.format(LR, MEMORY), histogram_freq=0, write_graph=True, write_images=True)


history = model.fit(x_train, y_train, validation_data=(x_test, y_test),  epochs=EPOCHS, verbose=2, callbacks = [tbCallBack])
model.evaluate(x_test, y_test, verbose=1)




plt.plot(history.history['acc'])#
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
if MEMORY == 1:
    plt.savefig('acc_lr={}.png'.format(KEY_SIZE, NUM_KEYS, LR, BANDWIDTH) )
else:
    plt.savefig('acc_momory=0&lr={}.png'.format(LR, BANDWIDTH))
plt.clf()

plt.plot(history.history['loss'])#
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
if MEMORY == 1:
    plt.savefig('loss_lr={}.png'.format(KEY_SIZE, NUM_KEYS, LR, BANDWIDTH) )
else:
    plt.savefig('loss_momory=0&lr={}.png'.format(LR, BANDWIDTH))   
plt.clf()





  
