import tensorflow as tf
import numpy as np
import math
from keras.models import Model
from keras.datasets import cifar10
import keras

#information from https://towardsdatascience.com/cifar-10-image-classification-in-tensorflow-5b501f7dc77c

#hyperparameters of the model
epochs = 10
batch_size = 16
keep_probability = 0.2 #dropout parameter
learning_rate = 0.0001
keep_prob=0.1
n_embedding = 10
keys_size=10
n_output = num_classes= 10
bandwith = 10000 #bandwidth

#initializing values matrix
values = np.vstack((np.repeat([[1,0,0,0,0,0,0,0,0,0]], keys_size, axis=0),
                    np.repeat([[0,1,0,0,0,0,0,0,0,0]], keys_size, axis=0),
                    np.repeat([[0,0,1,0,0,0,0,0,0,0]], keys_size, axis=0),
                    np.repeat([[0,0,0,1,0,0,0,0,0,0]], keys_size, axis=0),
                    np.repeat([[0,0,0,0,1,0,0,0,0,0]], keys_size, axis=0),
                    np.repeat([[0,0,0,0,0,1,0,0,0,0]], keys_size, axis=0),
                    np.repeat([[0,0,0,0,0,0,1,0,0,0]], keys_size, axis=0),
                    np.repeat([[0,0,0,0,0,0,0,1,0,0]], keys_size, axis=0),
                    np.repeat([[0,0,0,0,0,0,0,0,1,0]], keys_size, axis=0),
                    np.repeat([[0,0,0,0,0,0,0,0,0,1]], keys_size, axis=0)))
n_keys= values.shape[0]

#reading data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


print("Train size:", x_train.shape)
print("Test size:", x_test.shape)

x_train = x_train/255
x_test = y_test/255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#initializing placeholders
x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')
y =  tf.placeholder(tf.float32, shape=(None, 10), name='output_y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
V = tf.placeholder(tf.float32, shape = (n_keys, n_output))


#defining functions
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[ permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch_Y = shuffled_Y[ k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def sq_distance(A, B):

  row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
  row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

  row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
  row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

  return row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B

def kernel (A,B):
    
    d = sq_distance(A,B)/bandwith
    o = tf.reciprocal(d**2+1e-6)
    #o = tf.exp(tf.matmul(B,tf.transpose(A)))
    return o

def conv_net(x, keep_prob):

    conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 16], mean=0, stddev=0.08))
    conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 16, 32], mean=0, stddev=0.08))

    conv1 = tf.nn.conv2d(x, conv1_filter, strides=[1,1,1,1], padding='SAME')
    conv1 = tf.nn.relu(conv1)
    conv1_pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv1_bn = tf.layers.batch_normalization(conv1_pool)

    conv2 = tf.nn.conv2d(conv1_bn, conv2_filter, strides=[1,1,1,1], padding='SAME')
    conv2 = tf.nn.relu(conv2)
    conv2_pool = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    
    conv2_bn = tf.layers.batch_normalization(conv2_pool)
  
    flat = tf.contrib.layers.flatten(conv2_bn)  

    full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=64, activation_fn=tf.nn.relu)
    full1 = tf.nn.dropout(full1, keep_prob)
    full1 = tf.layers.batch_normalization(full1)
   
    full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=20, activation_fn=tf.nn.relu)
    full2 = tf.nn.dropout(full2, keep_prob)
    full2 = tf.layers.batch_normalization(full2)        
    
    #out = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=10, activation_fn=tf.nn.relu)
    out = full2

    return out


keys = tf.Variable(tf.random_uniform([n_keys, n_embedding],maxval=1), "keys")
embedding = conv_net(x, keep_prob)

K = kernel(keys, embedding)
KV =  tf.matmul(tf.transpose(K), V)
KV_ = tf.diag(tf.reshape(tf.reciprocal( tf.matmul(KV,tf.ones((n_output,1)))) , [-1]))
output = tf.matmul(KV_, KV)


# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')


with tf.Session() as sess:
    # Initializing the variables
   
    sess.run(tf.global_variables_initializer())

    #training the model
    for i in range(epochs):

        minibatches = random_mini_batches(x_train, y_train, batch_size, 1)

        for i, minibatch in enumerate(minibatches):
            batch_X, batch_Y = minibatch
            _, ks, emb, ker, o = sess.run([optimizer, keys, embedding, K, output], 
                                    feed_dict={x: batch_X, 
                                               y: batch_Y, 
                                               V: values,
                                               keep_prob: keep_probability})
            


            if(i%100==0):
                cost_, acc_ = sess.run([cost, accuracy], 
                                    feed_dict={x: x_test, 
                                               y: y_test, 
                                               V: values,
                                               keep_prob :keep_probability})                
                print("Minibatch ", i, " ...")
                print("Cost:", cost_)
                print("Acc:", acc_)
