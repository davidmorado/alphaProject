import tensorflow as tf
import pickle
import keras

#Definition of the Model
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
from keras.layers import Layer


class Varkeys(Layer):

    def __init__(self, keysize, dict_size, values, categories, **kwargs):
        self.output_dim = keysize
        self.initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=3, seed=None)
        #self.initializer = keras.initializers.random_uniform([dict_size, keysize],maxval=1)
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
        KV =  tf.matmul(tf.transpose(self.kernel(self.keys, x)), V)
        KV_ = tf.diag(tf.reshape(tf.reciprocal( tf.matmul(KV,tf.ones((self.categories,1)))) , [-1]))
        output = tf.matmul(KV_, KV)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.categories)

    def get_memory(self):
        return self.keys
    
    def sq_distance(self, A, B):
        # print('im in distance function')
        row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
        row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

        row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
        row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

        return row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B


    def kernel (self, A,B):
        # print('im in kernel function!!')
        d = self.sq_distance(A,B)/10000
        o = tf.reciprocal(d+1e-4)
        return o    

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
import keras

import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
input_shape = x_train.shape[1:]
num_classes = np.max(y_test)+1
num_samples = x_train.shape[0]

x_train = x_train/255
x_test = x_test/255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)




def CNN_keys(layers=[32, 64, 512], embedding_dim = 20, num_classes=10, n_keys= 100, V=[]):
        
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
    model.add(Varkeys(embedding_dim, n_keys, V, num_classes))


    return model


# def CNN(layers=[32, 64, 512], embedding_dim = 20, num_classes=10):

#     model = Sequential()

#     model.add(Conv2D(layers[0], (3, 3), padding='same',
#                     input_shape=x_train.shape[1:]))
#     model.add(Activation('relu'))
#     model.add(Conv2D(layers[0], (3, 3)))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))

#     model.add(Conv2D(layers[1], (3, 3), padding='same'))
#     model.add(Activation('relu'))
#     model.add(Conv2D(layers[1], (3, 3)))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))

#     model.add(Flatten())
#     model.add(Dense(layers[2]))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(embedding_dim))
#     model.add(Activation('sigmoid'))
#     model.add(BatchNormalization())
#     model.add(Dense(num_classes))
#     model.add(Activation('softmax'))  

#     return model


def fit_evaluate( model, x_train, y_train, x_test,  y_test, batch_size, epochs, lr):

    model.compile(loss=keras.losses.categorical_crossentropy,
                # optimizer=keras.optimizers.SGD(lr=0.1),
                optimizer = keras.optimizers.rmsprop(lr=lr, decay=1e-6),
                metrics=['accuracy'])

    history = model.fit(x_train, y_train,
            batch_size=  batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))

    memory = model.layers[-1].get_memory()
    
    val_acc = history.history['val_acc']
    acc = history.history['acc']
    loss = historyq.history['loss']
    val_loss = history.history['val_loss']
    
    scores = model.evaluate(x_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    return val_acc, acc, loss, val_loss, scores
    
n_output = 10
embedding_dim = 20
batch_size = 64
lr = 0.0001
epochs = 100
# this script varies the memory module size
numbers_of_keys_per_class = range(50, 10001, 50)

p = 1.0
#p = 0.1

for n_keys_per_class in numbers_of_keys_per_class:
  
    values = np.repeat(np.eye(10, dtype=int), n_keys_per_class, axis = 0)
    n_keys= values.shape[0]
    V = tf.constant(values, dtype=tf.float32, shape = (n_keys, n_output))

    print("Percentage of training =", p)
    idx = np.random.choice(num_samples, int(p*num_samples))
    x_train_ = x_train[idx,]
    y_train_ = y_train[idx,]

    print("CNN+Keys...")
    print("CNN with " + str(n_keys_per_class) + " keys per class.")
    model1 = CNN_keys(layers=[32, 64, 512], embedding_dim = 20, num_classes=10, n_keys= n_keys, V=V)
    results = fit_evaluate(model1, x_train_, y_train_, x_test, y_test, batch_size, epochs, lr)

    if n_keys_per_class < 200:
        plot_model(model1, to_file='results/CNN_' + str(n_keys_per_class) + '_keys_graph.png')
    
    filename = "results/CNN_" + str(n_keys_per_class) + "_keys.pkl"
    
    with open(filename, 'wb') as f:
      pickle.dump(results, f)
