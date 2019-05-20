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

x_train = x_train[:500]
x_test  = x_test[:100]
y_train = y_train[:500]
y_test  = y_test[:100]

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

history = model.fit(x_train, y_train, validation_data=(x_test, y_test),  epochs=4)

mem_layers = [49, 56, 64]
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