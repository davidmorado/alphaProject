import tensorflow as tf

from model import conv_net, conv_netV2, secondStage
from memory import Memory

from keras.datasets import cifar10
from keras.utils import to_categorical
import numpy as np
import sys

from random_mini_batches import random_mini_batches


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

# x_train = tf.random.shuffle(x_train,seed=1)
# y_train = tf.random.shuffle(y_train, seed=1)

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')
y = tf.placeholder(tf.float32, shape=(None, 10), name='output_y')

epochs = 50
batch_size = 32
learning_rate = 0.001

M = Memory(secondStage, batch_size=batch_size)
embeddings = conv_netV2(x)
logits = M.model(embeddings)

# Loss and Optimizer
cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=y))

original_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#optimizer = tf.contrib.estimator.clip_gradients_by_norm(original_optimizer, clip_norm=2.0)
train_op  = original_optimizer.minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

def training_step(session, optimizer, batch_features, batch_labels):
    _, h = session.run([optimizer, embeddings],
                feed_dict={
                    x: batch_features,
                    y: batch_labels
                })
    return h

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
    minibatches = minibatches[:-1]
    for i, minibatch in enumerate(minibatches):
        batch_X, batch_Y = minibatch
        hs = training_step(sess, train_op, batch_X, batch_Y)
        #_, hs = sess.run([embeddings, train_op], feed_dict={x: batch_X, y: batch_Y})
        M.write(hs, batch_Y)
            
    print('Epoch {:>2}, CIFAR-10 Batch:  '.format(epoch + 1), end='')
    #print_stats(sess, x_test, y_test)
    print_stats(sess, x_train, y_train)



def predict(x_, tfsession):
    # x_: [batchsize x 32 x 32 x 3]
    x = tf.placeholder(tf.float32, shape=(x_.shape[0], 32, 32, 3), name='input_x')
    embeddings = conv_net(x)
    yhats = M.predict(embeddings)
    with tf.Session() as sess:
        #print(sess.run(M.Keys))
        #print(sess.run(M.Values))
        sess.run(tf.global_variables_initializer())
        print('vars inited')
        print(sess.run(M.Keys))
        print(sess.run(M.Values))
        return sess.run(yhats, feed_dict={x:x_})

yhats = predict(x_test, sess)



print('HEREEEEE1')
print(np.array(yhats))
print(np.array(yhats).shape)
print('HEREEEEE2')
print(y)
print('HEREEEEE3')
correct_pred = tf.equal(tf.argmax(np.squeeze(np.array(yhats), axis=1), 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

print(sess.run(accuracy, feed_dict={y: y_test}))


sess.close()



