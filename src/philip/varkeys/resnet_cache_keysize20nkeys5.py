




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

KEY_SIZE = 20
NUM_KEYS = 5

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# Training parameters
batch_size = 32  # orig paper trained all networks with batch_size=128
epochs = 200
data_augmentation = True
num_classes = 10

# x_train = x_train[:200]
# x_test  = x_test[:50]
# y_train = y_train[:200]
# y_test  = y_test[:50]

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
        # KV =  tf.matmul(tf.transpose(K), V)
        # KV_ = tf.diag(tf.reshape(tf.reciprocal( tf.matmul(KV,tf.ones((n_output,1)))) , [-1]))
        # output = tf.matmul(KV_, KV)
        # return tf.nn.softmax(tf.matmul(tf.transpose(self.kernel(self.keys, x)), self.values))

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
        d = self.sq_distance(A,B)/10000
        o = tf.reciprocal(d+1e-4)
        return o  

def one_hot(length, i):
    return [1 if idx==i else 0 for idx in range(length)]



values = [one_hot(10, i) for i in range(num_classes)] * NUM_KEYS
n_keys= len(values)
n_output = y_train.shape[1]
V = tf.constant(values, dtype=tf.float32, shape = (n_keys, n_output))


base_model = ResNet50(weights=None,include_top=False, input_shape=input_shape)

x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(KEY_SIZE, activation='relu')(x)
predictions = Varkeys(KEY_SIZE, n_keys, V, num_classes)(x)
# and a logistic layer -- 10 classes for CIFAR10
#predictions = Dense(num_classes, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)



model.compile(
    optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(x_train, y_train, validation_data=(x_test, y_test),  epochs=15)
model.evaluate(x_test, y_test)

np.save('accuracy', np.array(history.history['acc']))
np.save('val_accuracy', np.array(history.history['val_acc']))

plt.plot(history.history['acc'])#
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('learningcurve_keysize={}&nkeys={}'.format(KEY_SIZE, NUM_KEYS) )






  
