


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
from keras import backend as K

import sys

# set default parameters
KEY_SIZE = 100  # keysize (= embedding size)
NUM_KEYS = 1000 # number of keys per class
LR=0.0001       # learning rate
BANDWIDTH = 10000 # bandwith parameter
MEMORY = 1

NUM_CLASSES = 10
EPOCHS = 320




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



base_model = ResNet50(weights='imagenet',include_top=False, input_shape=input_shape)

# freeze pretrained model
for layer in base_model.layers[1:-2]:
    layer.trainable=False

x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

embedding = K.function(model.input, x.output)
