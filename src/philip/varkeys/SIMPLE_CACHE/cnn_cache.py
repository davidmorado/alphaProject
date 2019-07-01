




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
EPOCHS = 50
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

x_train = x_train[:10000]
x_test  = x_test[:1000]
y_train = y_train[:10000]
y_test  = y_test[:1000]

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)



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



model = CNN(embedding_dim = KEY_SIZE, num_classes=NUM_CLASSES)

print(model.summary())
sys.exit(0)

model.compile(loss=keras.losses.categorical_crossentropy,
            # optimizer=keras.optimizers.SGD(lr=0.1),
            optimizer = keras.optimizers.Adam(lr=LR, decay=1e-6),
            metrics=['accuracy'])


# tensorboard
tbCallBack = keras.callbacks.TensorBoard(log_dir='tb_logs/lr={}&memory={}'.format(LR, MEMORY), histogram_freq=0, write_graph=True, write_images=True)


history = model.fit(x_train, y_train, validation_data=(x_test, y_test),  epochs=EPOCHS, verbose=2, callbacks = [tbCallBack], batch_size=batch_size)
model.evaluate(x_test, y_test, verbose=1)


mem_layers = [1, 2]
output_list = []
for i in range(len(mem_layers)):
    output_list.append(model.layers[mem_layers[i]].output)

# extra memory model:
mem = Model( inputs=model.input, outputs=output_list )

data_frac = 1
cache_indices = np.random.permutation(np.arange(x_train.shape[0]))#[:np.int(data_frac)]
x_train = x_train[cache_indices, :, :, :]
y_train = y_train[cache_indices, :]

memkeys_list = mem.predict(x_train)

#print('memkeys_list shape: ', memkeys_list.shape) # (100, 10)
mem_keys = np.reshape(memkeys_list[0],(x_train.shape[0],-1))
#mem_keys = memkeys_list.reshape((x_train.shape[0],-1))
for i in range(len(mem_layers)-1):
    mem_keys = np.concatenate((mem_keys, np.reshape(memkeys_list[i+1],(x_train.shape[0],-1))),axis=1)

# Memory values
mem_vals = y_train


# Pass items thru memory
testmem_list = mem.predict(x_test)
test_mem = np.reshape(testmem_list[0],(x_test.shape[0],-1))
for i in range(len(mem_layers)-1):
    test_mem = np.concatenate((test_mem, np.reshape(testmem_list[i+1],(x_test.shape[0],-1))),axis=1)


# Normalize keys and query
query = test_mem / np.sqrt( np.tile(np.sum(test_mem**2, axis=1, keepdims=1),(1,test_mem.shape[1])) )
key = mem_keys / np.sqrt( np.tile(np.sum(mem_keys**2, axis=1, keepdims=1),(1,mem_keys.shape[1])) )

theta = 0.5   
similarities = np.exp( theta * np.dot(query, key.T) )
p_mem = np.matmul(similarities, mem_vals)
p_mem = p_mem / np.repeat(np.sum(p_mem, axis=1, keepdims=True), num_classes, axis=1)

p_model = model.predict(x_test)

lmbd  = 0.9
p_combined = (1.0-lmbd) * p_model + lmbd * p_mem

pred_combined = np.argmax(p_combined, axis=1)
y_test_int = np.argmax(y_test, axis=1)
test_acc = np.mean(pred_combined==y_test_int)

print('Mem. shape:', mem_keys.shape)
print('Mem. accuracy:', test_acc)

loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
print('model loss:', loss)
print('model accuracy:', accuracy)


np.save('accuracy', np.array(history.history['acc']))
np.save('val_accuracy', np.array(history.history['val_acc']))

plt.plot(history.history['acc'])#
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('learningcurve')