import tensorflow as tf

#Definition of the Model
from keras.models import Model
from keras.layers.normalization import BatchNormalization

from keras.layers import Layer
import keras

class Varkeys(Layer):

    def __init__(self, keysize, dict_size, values, categories, **kwargs):
        self.output_dim = keysize
        self.initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=None)
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

    
    def sq_distance(self, A, B):
        print('im in distance function')
        row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
        row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

        row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
        row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

        return row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B


    def kernel (self, A,B):
        print('im in kernel function!!')
        d = self.sq_distance(A,B)/300000
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
input_shape = x_train.shape[1:]
num_classes = np.max(y_test)+1
num_samples = x_train.shape[0]

x_train = x_train/255
x_test = x_test/255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

n_output = 10
embedding_dim = 100
n_keys_per_class = 50
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
n_keys= values.shape[0]
V = tf.constant(values, dtype=tf.float32, shape = (n_keys, n_output))


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
    model.add(Varkeys(embedding_dim, n_keys, V, num_classes))
    


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

def custom_loss(layer, sigma, custom=1):

    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true,y_pred):

      if(custom==1):
        return keras.losses.categorical_crossentropy(y_true=y_true, y_pred=y_pred)+sigma*tf.reduce_mean(layer.kernel(layer.keys, layer.keys))# + sigma*tf.reduce_mean(layer.kernel(layer.keys, layer.keys) , axis=-1)
      else:
        return keras.losses.categorical_crossentropy(y_true=y_true, y_pred=y_pred)
   
    # Return a function
    return loss
    


def fit_evaluate( model, x_train, y_train, x_test,  y_test, batch_size, epochs, lr, sigma=0.001, custom=1):

    model.compile(loss=custom_loss(model.layers[-1], sigma, custom),#keras.losses.categorical_crossentropy,
                # optimizer=keras.optimizers.SGD(lr=0.1),
                optimizer = keras.optimizers.rmsprop(lr=lr, decay=1e-6),
                metrics=['accuracy'])

    model.fit(x_train, y_train,
            batch_size=  batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))

    scores = model.evaluate(x_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    return model


batch_size = 64
lr = 0.00038
epochs = 30
sigma=0.00016


#print("CNN+Keys...")
#model1 = CNN_keys(layers=[32, 64, 512], embedding_dim = embedding_dim, num_classes=10, n_keys= n_keys, V=V)
#fit_evaluate( model1, x_train, y_train, x_test, y_test, batch_size, epochs, lr, sigma=sigma, custom=0)


print("CNN...")
model2 = CNN(layers=[32, 64, 512], embedding_dim = embedding_dim, num_classes=10)
model2= fit_evaluate( model2, x_train, y_train, x_test, y_test, batch_size, epochs, lr, sigma=sigma, custom=0)


#copying the first layers of the model CNN and adding varkyes
model_aux = Sequential()

for l in model2.layers[:-2]:
  model_aux.add(l)

model3 = keras.models.clone_model(model_aux)
model3.set_weights(model_aux.get_weights())
model3.add(Varkeys(embedding_dim, n_keys, V, num_classes))
fit_evaluate( model3, x_train, y_train, x_test, y_test, batch_size, epochs, lr, sigma=sigma, custom=0)

#further training of the CNN model
fit_evaluate( model2, x_train, y_train, x_test, y_test, batch_size, epochs, lr, sigma=sigma, custom=0)