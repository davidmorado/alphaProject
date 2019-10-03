import tensorflow as tf

from model import conv_net, secondStage
from memory import Memory

from keras.datasets import cifar10
from keras.utils import to_categorical
import numpy as np

# Data Loading and Preprocessing
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
input_shape = x_train.shape[1:]
num_classes = np.max(y_test)+1
num_samples = x_train.shape[0]

x_train = x_train/255
x_test = x_test/255
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

x_train = x_train[:100]
y_train = y_train[:100]
x_test = x_test[:100]
y_test = y_test[:100]

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')
y = tf.placeholder(tf.float32, shape=(None, 10), name='output_y')

epochs = 1
batch_size = 32
learning_rate = 0.001


M = Memory(secondStage, batch_size=batch_size)

# import sys
# sys.exit(0)


embeddings = conv_net(x)
logits = M.model(embeddings)

# Loss and Optimizer
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

def split_in_batches(x_train, y_train, bs):
    N = len(x_train)
    batches = int(N/bs)
    batch_features = [x_train[i:i+bs] for i in range(batches)]
    batch_labels = [y_train[i:i+bs] for i in range(batches)]
    return batch_features, batch_labels

def training_step(session, optimizer, batch_features, batch_labels):
    _, h = session.run([optimizer, embeddings],
                feed_dict={
                    x: batch_features,
                    y: batch_labels
                })
    return h

def print_stats(session, valid_features, valid_labels, cost, accuracy):
    valid_acc = sess.run(accuracy, 
                         feed_dict={
                             x: valid_features,
                             y: valid_labels
                         })
    
    print('Validation Accuracy: {:.6f}'.format(valid_acc))

with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        batch_i = 0
        for batch_features, batch_labels in zip(*split_in_batches(x_train, y_train, bs=batch_size)):
            hs = training_step(sess, optimizer, batch_features, batch_labels)
            M.write(hs, batch_labels)
            batch_i+=1    
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, x_test, y_test, cost, accuracy)

def predict(x_):
    # x_: [batchsize x 32 x 32 x 3]
    x = tf.placeholder(tf.float32, shape=(x_.shape[0], 32, 32, 3), name='input_x')
    embeddings = conv_net(x)
    # print(embeddings)
    yhats = M.predict(embeddings)
    with tf.Session() as sess:
        return sess.run(yhats, feed_dict={x:x_})


yhats = predict(x_test)
correct_pred = tf.equal(tf.argmax(yhats, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

print(sess.run(accuracy, feed_dict={y: y_test}))




