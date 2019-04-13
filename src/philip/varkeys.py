# -*- coding: utf-8 -*-
"""
@author: Philip Kurzendörfer

Code implementing Neural Differential Dictionaries with variable keys
https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria
"""



import tensorflow as tf
from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv1D
from tensorflow.keras.layers import Flatten, Add
#from tensorflow.keras.layers.embeddings import Embedding
from tensorflow.keras.layers import Embedding
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda
import tensorflow.keras
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers

#Importing of training photos of the data set
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Definition of the additional layers
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D

#Import Libraries
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet import preprocess_input

from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
# from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import BatchNormalization

#Import of MobileNet repositories
from tensorflow.keras.applications import MobileNet
mobilenet=MobileNet(weights='imagenet') 


#Definition of the Model
from tensorflow.keras.models import Model


from tensorflow.keras.layers import Layer

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
        return tf.matmul(tf.transpose(self.kernel(self.keys, x)), self.values)

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
        d = self.sq_distance(A,B)
        o = tf.reciprocal(d+1e-4)
        return o    




values = [[0,1]]*10 + [[1, 0]]*2

n_keys= len(values)

n_output = 2
embedding_dim = 64

V = tf.constant(values, dtype=tf.float32, shape = (n_keys, n_output))


# define the model:

# base_model=MobileNet(weights='imagenet', include_top=False) 
# x=base_model.output
# x=GlobalAveragePooling2D()(x)
# x=Dense(1024,activation='relu')(x)
# x=Dense(512,activation='relu')(x)
# #x=Dense(100,activation='relu')(x)
# #K = Varkeys(embedding_dim, n_keys, V, 2)(x)
# preds=Dense(2,activation='softmax')(x)
# model=Model(inputs=base_model.input,outputs=preds)
# model.summary()

# #Determining the layers for additional training
# for layer in model.layers[:-5]:
#     layer.trainable=False


# kaggle model with varkey layer


# custom model
#Inializing CNN
model = Sequential()

#Adding 1st Convolution Layer
model.add(Convolution2D(filters=32, kernel_size=(3,3), strides=(1,1),input_shape=(32,32,3), activation='relu'))
#Adding 1st MaxPooling Layer to reduce the size of Features
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

#Adding Batch Normalization for Higher Learnig Rate
model.add(BatchNormalization())
#Adding Dropout Layer to eliminate Overfitting
model.add(Dropout(0.2))

#Adding 2nd Convolution Layer
model.add(Convolution2D(filters=32, kernel_size=(3,3), activation='relu'))
#Adding 2nd MaxPooling Layer
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

#Adding Flatten Layer to convert 2D matrix into an array
model.add(Flatten())

#Adding 1st Fully Connected Layer
model.add(Dense(units=64, activation='relu'))

#model.add(Varkeys(embedding_dim, n_keys, V, 2))
#Adding Fully Connected Output Layer
model.add(Dense(2,activation='softmax'))


# lead the data
datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.5)
train_data_generator = datagen.flow_from_directory(
    'C:\\daten\\python_envs\\deeplearning\\alpha5-old\\alpha5\\development_code\\philip\\VarKeys\\data\\cell_images',
    target_size=(32,32),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=True,
    subset="training"
    
)

test_data_generator = datagen.flow_from_directory(
    #'data\\cell_images\\',
    'C:\\daten\\python_envs\\deeplearning\\alpha5-old\\alpha5\\development_code\\philip\\VarKeys\\data\\cell_images',
    target_size=(32,32),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=True,
    subset="validation"
    
)


#Model Compilation
model.compile(
    optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


#Model Training
history = model.fit_generator(
    generator=train_data_generator,
    steps_per_epoch=train_data_generator.n/train_data_generator.batch_size,
    epochs=1,
    #validation_data=test_data_generator,
    #validation_steps=test_data_generator.n/test_data_generator.batch_size,
)

# https://keras.io/models/sequential/


print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['accuracy'])#
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()



