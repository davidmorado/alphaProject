import keras
import tensorflow as tf  

import numpy as np
    
from keras.datasets import cifar10
#from tensorflow.keras.applications import ResNet50


from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from skimage.transform import resize

import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# Training parameters
batch_size = 32  # orig paper trained all networks with batch_size=128
epochs = 200
data_augmentation = True
num_classes = 10

x_train = x_train[:200]
x_test  = x_test[:50]
y_train = y_train[:200]
y_test  = y_test[:50]

# # Resize image arrays
# x_train = resize_image_arr(x_train)
# x_test = resize_image_arr(x_test)

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


base_model = ResNet50(weights='imagenet',include_top=False, input_shape=input_shape)

x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(512, activation='relu')(x)
# and a logistic layer -- 10 classes for CIFAR10
predictions = Dense(num_classes, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)



model.compile(
    optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(x_train, y_train, validation_data=(x_test, y_test),  epochs=1)
model.evaluate(x_test, y_test)

np.save('accuracy', np.array(history.history['acc']))
np.save('val_accuracy', np.array(history.history['val_acc']))

plt.plot(history.history['acc'])#
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('learningcurve')