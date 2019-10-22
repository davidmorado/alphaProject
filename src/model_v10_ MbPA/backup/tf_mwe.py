from __future__ import absolute_import, division, print_function, unicode_literals
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
import numpy as np
from random_mini_batches import *
from keras.datasets import cifar10
from keras.utils import to_categorical


def conv_netV2(x, embedding_size=50):
    conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 32], mean=0, stddev=0.1))
    conv1_filter2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 32], mean=0, stddev=0.1))
    print(conv1_filter2)
    print(x)

    conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 64], mean=0, stddev=0.1))
    conv2_filter2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], mean=0, stddev=0.1))
   

    # conv 1 block
    conv1 = tf.nn.conv2d(x, conv1_filter, strides=[1,1,1,1], padding='SAME')
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.conv2d(conv1, conv1_filter2, strides=[1,1,1,1], padding='VALID')
    conv1 = tf.nn.relu(conv1)
    conv1_pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    conv1_pool = tf.nn.dropout(conv1_pool, keep_prob=0.75)

    # conv 2 block
    conv2 = tf.nn.conv2d(conv1_pool, conv2_filter, strides=[1,1,1,1], padding='SAME')
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.conv2d(conv2, conv2_filter2, strides=[1,1,1,1], padding='VALID')
    conv2 = tf.nn.relu(conv2)
    conv2_pool = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    conv2_pool = tf.nn.dropout(conv2_pool, keep_prob=0.75)

    # dense block
    flat = tf.contrib.layers.flatten(conv2_pool)  
    full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=512, activation_fn=tf.nn.relu)
    full1 = tf.nn.dropout(full1, keep_prob=0.5) 
    full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=embedding_size, activation_fn=tf.nn.sigmoid)
    full2 = tf.layers.batch_normalization(full2)
    full2 = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=10, activation_fn=None)
    
    return full2 # shape = (batch_size, embeddingsize) 




# Data Loading and Preprocessing
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
input_shape = x_train.shape[1:]
num_classes = np.max(y_test)+1
num_samples = x_train.shape[0]

x_train = x_train/255
x_test = x_test/255
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

x_train = x_train[:300]
y_train = y_train[:300]
x_test = x_test[:20]
y_test = y_test[:20]

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')
y = tf.placeholder(tf.float32, shape=(None, 10), name='output_y')

epochs = 200
batch_size = 32
learning_rate = 0.001

logits = conv_netV2(x)

# Loss and Optimizer
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=y))

original_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#optimizer = tf.contrib.estimator.clip_gradients_by_norm(original_optimizer, clip_norm=2.0)
train_op  = original_optimizer.minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

def print_stats(session, valid_features, valid_labels):
    valid_acc, valid_cost = session.run([accuracy, cost],
                         feed_dict={
                             x: valid_features,
                             y: valid_labels
                         })
    
    print('Validation Accuracy: {:.6f} \t Cost: {:.6f}'.format(valid_acc, valid_cost))

#with tf.Session() as sess:
sess = tf.Session()
# Initializing the variables
sess.run(tf.global_variables_initializer())

# Training cycle
for epoch in range(epochs):
    # Loop over all batches
    minibatches = random_mini_batches(x_train, y_train, batch_size, 1)
    for i, minibatch in enumerate(minibatches):
        batch_X, batch_Y = minibatch
        sess.run(train_op, feed_dict={x: batch_X,y: batch_Y})
            
    print('Epoch {:>2}, CIFAR-10 Batch:  '.format(epoch + 1), end='')
    #print_stats(sess, x_test, y_test)
    print_stats(sess, x_train, y_train)
