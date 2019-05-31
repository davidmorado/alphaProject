import tensorflow as tf
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
import numpy as np
from keras.layers import Layer
import keras

class Varkeys(Layer):

    def __init__(self, keysize, dict_size, values, categories, bandwidth, **kwargs):
        self.output_dim = keysize
        self.initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=None)
        #self.initializer = keras.initializers.random_uniform([dict_size, keysize],maxval=1)
        self.values = values
        self.categories = categories
        self.keysize = keysize 
        self.dict_size = dict_size
        self.bandwidth = bandwidth
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

    
    def sq_distance(self, A, B):
        print('im in distance function')
        row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
        row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

        row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
        row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

        return row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B


    def kernel (self, A,B):
        print('im in kernel function!!')
        d = self.sq_distance(A,B)/self.bandwidth
        o = tf.reciprocal(d+1e-4)
        #o = tf.exp(-d/10)
        return o
    
    def kernel_cos(self, A,B):
      
        normalize_A = tf.nn.l2_normalize(A,1)        
        normalize_B = tf.nn.l2_normalize(B,1)
        cossim = tf.matmul(normalize_B, tf.transpose(normalize_A))
        return tf.transpose(cossim)
      
    def kernel_gauss(self, A,B):
        d = self.sq_distance(A,B)
        o = tf.exp(-(d)/100)
        return o

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
import keras

import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, x_val, y_train, y_val = train_test_split( x_train, y_train, test_size=0.33, random_state=42)

print("x_val", x_val.shape)

input_shape = x_train.shape[1:]
num_classes = np.max(y_test)+1
num_samples = x_train.shape[0]

x_train = x_train/255
x_test = x_test/255
x_val = x_val/255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

def create_dict(n_keys_per_class):
  
  values = np.vstack((np.repeat([[1,0,0,0,0,0,0,0,0,0]], n_keys_per_class, axis=0),
                    np.repeat([[0,1,0,0,0,0,0,0,0,0]], n_keys_per_class, axis=0),
                    np.repeat([[0,0,1,0,0,0,0,0,0,0]], n_keys_per_class, axis=0),
                    np.repeat([[0,0,0,1,0,0,0,0,0,0]], n_keys_per_class, axis=0),
                    np.repeat([[0,0,0,0,1,0,0,0,0,0]], n_keys_per_class, axis=0),
                    np.repeat([[0,0,0,0,0,1,0,0,0,0]], n_keys_per_class, axis=0),
                    np.repeat([[0,0,0,0,0,0,1,0,0,0]], n_keys_per_class, axis=0),
                    np.repeat([[0,0,0,0,0,0,0,1,0,0]], n_keys_per_class, axis=0),
                    np.repeat([[0,0,0,0,0,0,0,0,1,0]], n_keys_per_class, axis=0),
                    np.repeat([[0,0,0,0,0,0,0,0,0,1]], n_keys_per_class, axis=0)))
  return values

def CNN_keys(layers=[32, 64, 512], embedding_dim = 20, num_classes=10, n_keys= 100, bandwidth=10000, V=[]):
        
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
    model.add(Varkeys(embedding_dim, n_keys, V, num_classes, bandwidth))

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


def fit_evaluate( model, x_train, y_train, x_test,  y_test, batch_size, epochs, lr):

    model.compile(loss=keras.losses.categorical_crossentropy,
                # optimizer=keras.optimizers.SGD(lr=0.1),
                optimizer = keras.optimizers.rmsprop(lr=lr, decay=1e-6),
                metrics=['accuracy'])

    model.fit(x_train, y_train,
            batch_size=  batch_size,
            epochs=epochs,
            verbose=0)

    scores_train = model.evaluate(x_train, y_train, verbose=0)
    scores_test = model.evaluate(x_test, y_test, verbose=0)
    print("Train \n%s: %.2f%%" % (model.metrics_names[1], scores_train[1]*100))
    print("Val \n%s: %.2f%%" % (model.metrics_names[1], scores_test[1]*100))

    return scores_train, scores_test


batch_size = 64
epochs = 100
n_output = 10
n_hype_try = 50


def scale(value, min_v, max_v):
  return min_v + (value * (max_v - min_v))

lim_lr =(-5, -1)
lim_embedding = (1, 4)
lim_n_keys = (1, 3)
lim_bandwidth = (2, 6)
best_score1 = [0, 0, 0, 0]
best_score2 = [0, 0, 0, 0]
hyp_list = []
perf_list = []

for i in range(50):

  l = 10**scale(np.random.rand(), lim_lr[0], lim_lr[1])
  ed = int(10**scale(np.random.rand(), lim_embedding[0], lim_embedding[1]))
  nk = int(10**scale(np.random.rand(), lim_n_keys[0], lim_n_keys[1]))
  b =  int(10**scale(np.random.rand(), lim_bandwidth[0], lim_bandwidth[1]))
  
  print("Parameters ... B: ", b," L: ", l, " ED: ",ed, "NK:", nk )
  values = create_dict(nk)
  n_keys= values.shape[0]
  V = tf.constant(values, dtype=tf.float32, shape = (n_keys, n_output))
          
  print("CNN+Keys...")
  model1 = CNN_keys(layers=[32, 64, 512], embedding_dim = ed, num_classes=10, n_keys= n_keys, bandwidth=b, V=V)
  scores_train1, scores_test1 = fit_evaluate( model1, x_train, y_train, x_val, y_val, batch_size, epochs, l)

  print("CNN...")
  model2 = CNN(layers=[32, 64, 512], embedding_dim = ed, num_classes=10)
  scores_train2, scores_test2 = fit_evaluate( model2, x_train, y_train, x_val, y_val, batch_size, epochs, l)

  if(scores_test1[1]>best_score1[1]):
      best_hyperparameters1 = (b, l, ed, nk)
      best_score1[0] = scores_train1[1]
      best_score1[1] = scores_test1[1]
      best_score1[2] = scores_train2[1]
      best_score1[3] = scores_test2[1]

  if(scores_test2[1]>best_score2[3]):
      best_hyperparameters2 = (b, l, ed, nk)
      best_score2[0] = scores_train1[1]
      best_score2[1] = scores_test1[1]
      best_score2[2] = scores_train2[1]
      best_score2[3] = scores_test2[1]

  hyp_list.append([b, l, ed, nk])
  perf_list.append([scores_train1, scores_test1, scores_train2, scores_test2])

print("Best score:", best_score1)
print("Best hyperparameters:", best_hyperparameters1)

print("Best score:", best_score2)
print("Best hyperparameters:", best_hyperparameters2)

import pickle

with open("performance", 'wb') as f:
    pickle.dump([hyp_list, perf_list], f)