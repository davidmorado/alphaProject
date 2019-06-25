import tensorflow as tf
import keras

#Definition of the Model
#from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers import Layer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
import numpy as np

from keras.callbacks import Callback
#from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

class Varkeys(Layer):

    def __init__(self, keysize, dict_size, values, categories, bandwidth, **kwargs):
        
        self.output_dim = keysize  #20
        self.initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=None)
        #self.initializer = keras.initializers.random_uniform([dict_size, keysize],maxval=1)
        self.values = values
        self.categories = categories
        self.keysize = keysize 
        self.dict_size = dict_size  #50
        self.bandwidth = bandwidth
        
        super(Varkeys, self).__init__(**kwargs)  # gain access to inherited methods – from a parent or sibling class

    def build(self, input_shape):    #Keys matrix is created
        # Create a trainable weight variable for this layer.
        self.keys = self.add_weight(name='keys',    #50 * 20
                                      shape=(self.dict_size, self.keysize),
                                      initializer=self.initializer,
                                      trainable=True)
        
        super(Varkeys, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        KV =  tf.matmul(tf.transpose(self.kernel(self.keys, x)), V)  #20 * 10 sanırım
        #print("KV",tf.Session().run(KV))
        
        print(KV)  # ? , 20
        '''init = tf.global_variables_initializer()
        #run the graph
        with tf.Session() as sess:
            sess.run(init) #execute init
            print (sess.run(KV))'''
        
        
        KV_ = tf.diag(tf.reshape(tf.reciprocal( tf.matmul(KV,tf.ones((self.categories,1)))) , [-1]))
        print(KV_)  # 1 , 20  sanırım                                    (20 * 10 )*(10 * 1) = 20*10    
        
        
        output = tf.matmul(KV_, KV)
        return output

    def compute_output_shape(self, input_shape):  #50000 - 10
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
        
        #d = self.sq_distance(A,B)/bw
        #o = tf.reciprocal(d+(1/bw))  # 1 / (Distance/10000)+(1/10000)
        
        
        d = self.sq_distance(A,B)/self.bandwidth #Distance / Bandwidth
        o = tf.reciprocal(d+1e-4)
        #o = tf.exp(-d/10)
        return o    


def sq_distance(A, B):
        print('im in distance function')
        row_norms_A = tf.reduce_sum(tf.square(A), axis=1)  #axis = 1 demek rowwise topla demek
        #print(row_norms_A.eval())
        row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector. #1*1

        row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
        row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.    #1*1

        return row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B

'''a = [2,5,6]
b = [5,7,3]
sq_distance(a, b)

A=tf.reshape(tf.Variable([2,5,6]) , [1, 3]);A
B=tf.reshape(tf.Variable([5,7,3]) , [1, 3]);B
#B=tf.Variable([5,7,3])

D = sq_distance(A, B)

init_op = tf.global_variables_initializer()

#run the graph
with tf.Session() as sess:
    sess.run(init_op) #execute init_op
    #print the random values that we sample
    print (sess.run(D))'''




# Decrease number of class to 3 

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# To subset the data
'''selected = [0,1,2]
X_train=np.where(np.any(X_train==selected, axis=1))
X_test=np.where(np.any(X_test==selected, axis=1))
y_train=np.where(np.any(y_train==selected, axis=1))
y_test=np.where(np.any(y_test==selected, axis=1))'''


#np.ndim(index[0])
#len(index[0])

x_train.shape   #(50000, 32, 32, 3)
x_test.shape
x_train[:1]
y_train[:3]    
y_train.shape   
y_test.shape

input_shape = x_train.shape[1:]   #(32, 32, 3)
num_classes = np.max(y_test)+1 ;num_classes   #10 classes
num_samples = x_train.shape[0] ;num_samples   #50000

# Normalization of values
x_train = x_train/255
x_test = x_test/255

# Convert labels to category
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

y_train.shape

n_output = 10 # number of classese
embedding_dim = 20
n_keys_per_class = 5

# values matrisi number_of_per_class kadar 1-0 matrisi yaratıp aşağı doğru , farklı 
# key_per_class=5 * class number=10
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


values.shape  ##(50, 10)
n_keys= values.shape[0]  #50, n_keys is number of cell

V = tf.constant(values, dtype=tf.float32, shape = (n_keys, n_output))
#print(tf.Session().run(V)) #V =  50*10

# layers : filter numbers
def CNN_keys(layers=[32, 64, 512], embedding_dim = 20, num_classes=10, n_keys= n_keys, bandwidth=1000, V=[]): #n_keys = 50
        
    model = Sequential()

    model.add(Conv2D(layers[0], (3, 3), padding='same',  # (32 tane convolution yani, 32 tane filter çalışıyor)
                    input_shape=x_train.shape[1:]))      # (32, 32, 3)
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
    model.add(Dense(layers[2]))   #
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(embedding_dim))   #20 tane FCLa bağlıyor
    model.add(Activation('sigmoid'))
    
    model.add(BatchNormalization())  #ReLus value can be too high
    
    # 
    model.add(Varkeys(embedding_dim, n_keys, V, num_classes,bandwidth))  #20 , 50 , values , 10


    return model

'''for layer in model.layers:
    print(layer.output_shape)'''



def CNN(layers=[32, 64, 512], embedding_dim = 20, num_classes=10):

    model = Sequential()

    model.add(Conv2D(layers[0], (3, 3), padding='same',  # (None, 32, 32, 32)
                    input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(layers[0], (3, 3)))       # (None, 30, 30, 32)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # (None, 15, 15, 32)
    model.add(Dropout(0.25))

    model.add(Conv2D(layers[1], (3, 3), padding='same'))  # (None, 15, 15, 64)
    model.add(Activation('relu'))
    model.add(Conv2D(layers[1], (3, 3)))                  # (None, 13, 13, 64)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))             # (None, 6, 6, 64)
    model.add(Dropout(0.25))

    model.add(Flatten())                                   # (None, 2304)
    model.add(Dense(layers[2]))                           # (None, 512)
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(embedding_dim))                       # (None, 20)
    # sigmoiddi relu yaptım
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(num_classes))                         # (None, 10)
    model.add(Activation('softmax'))  

    return model

# To visualize layers shape
'''for layer in model.layers:
    print(layer.output_shape)'''
    
def fit_evaluate( model, x_train, y_train, x_test,  y_test, batch_size, epochs, lr):

    model.compile(loss=keras.losses.categorical_crossentropy,
                # optimizer=keras.optimizers.SGD(lr=0.1),
                #optimizer = keras.optimizers.rmsprop(lr=lr, decay=1e-6),
                #optimizer = keras.optimizers.Adam(lr=lr, decay=1e-6),
                optimizer = keras.optimizers.Adadelta(lr=1.0,rho=0.95,decay=0.0),
                metrics=['accuracy'])
    
    
    callbacks = [
    #EarlyStoppingByLossVal(monitor='val_loss', value=0.00001, verbose=1)]
    EarlyStopping(monitor='val_loss', patience=10, verbose=0),
    ModelCheckpoint(filepath="asd",monitor='val_loss', save_best_only=True, verbose=0)]
    
    model.fit(x_train, y_train,
            batch_size=  batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))

    scores = model.evaluate(x_test, y_test)
    print("\n%s: %.2f%%--------" % (model.metrics_names[1], scores[1]*100))


batch_size = 64
lr = 0.0004
epochs = 30
bandwidth = 1000

#perc_data = [0.1, 0.2, 0.4, 0.8,1.0]
perc_data=[0.1]
p = 0.1
for p in perc_data:

    print("Pecentage of training =", p)
    #x_train = np
    idx = np.random.choice(num_samples, int(p*num_samples)) # 50000 indexten perc'ına göre seçiyor
    x_train_ = x_train[idx,]
    y_train_ = y_train[idx,]

    print("CNN+Keys...")      #height , width , stack
    model1 = CNN_keys(layers=[32, 64, 512], embedding_dim = 20, num_classes=10, n_keys= n_keys, V=V) #n_keys = 50 -> 5 * 10 
    fit_evaluate( model1, x_train_, y_train_, x_test, y_test, batch_size, epochs, lr)


    print("CNN...")
    model2 = CNN(layers=[32, 64, 512], embedding_dim = 20, num_classes=10)
    fit_evaluate( model2, x_train_, y_train_, x_test, y_test, batch_size, epochs, lr)
    
num_samples  
 
    
# Python program to illustrate   
# *args for variable number of arguments 
def myFun(*argv):   
    for arg in argv:  
        print (arg) 
    
myFun('Hello', 'Welcome', 'to', 'GeeksforGeeks') 

# Python program to illustrate   
# *kargs for variable number of keyword arguments 
def myFun(**kwargs):  
    for key, value in kwargs.items(): 
        print ("%s == %s" %(key, value)) 
  
# Driver code 
#k = (first ='Geeks', mid ='for', last='Geeks')

'''k =	{
  "first": "Geeks",
  "mid": "for",
  "last": "Geeks"
}

myFun(k)'''

myFun(first ='Geeks', mid ='for', last='Geeks')

# 1.st learn keys then keep keys fixed then learn NN network






