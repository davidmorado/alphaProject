import tensorflow as tf

from model import conv_net, conv_netV2, secondStage, SecondStage
from memory import Memory

from keras.datasets import cifar10
from keras.utils import to_categorical
import numpy as np
import sys

from random_mini_batches import random_mini_batches


# hyperparameters
epochs = 100
batch_size = 32
learning_rate = 0.001
nearest_neighbors = 50
validation_freq = 10


# read hyperparameters from command line arguments and overwrite default ones
hp_dict_str = sys.argv[1]
import yaml
hp_dict = yaml.load(hp_dict_str)

#hp_dict = ast.literal_eval(hp_dict_str)
for key,val in hp_dict.items():
    exec(key + '=val')
print('nearest_neighbors: ',nearest_neighbors)
print('update_period: ',update_period)




# Data Loading and Preprocessing
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_val, x_test, y_train, y_val, y_test = get_dataset(dataset, split_ratio)

input_shape = x_train.shape[1:]
num_classes = np.max(y_val)+1
num_samples = x_train.shape[0]

x_train = x_train/255
x_val = x_val/255
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)

x_train = x_train[:500].astype(np.float32)
y_train = y_train[:500].astype(np.float32)
x_val = x_val[:100].astype(np.float32)
y_val = y_val[:100].astype(np.float32)

# x_train = tf.random.shuffle(x_train,seed=1)
# y_train = tf.random.shuffle(y_train, seed=1)

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')
y = tf.placeholder(tf.float32, shape=(None, 10), name='output_y')



sess = tf.Session()
#sess.run(tf.global_variables_initializer())

M = Memory(SecondStage(), batch_size=batch_size, session=sess)
M.initialize()
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


def predict(x_, tfsession):
    # x_: [batchsize x 32 x 32 x 3]
    print(x_.shape)
    print(x_.dtype)
    #x_in = tf.placeholder(tf.float32, shape=(x_.shape[0], 32, 32, 3), name='x_in', )
    hs_ = tfsession.run(embeddings, feed_dict={x:x_})
    yhats = M.predict(hs_)
    
    #sess.run(tf.global_variables_initializer())
    print('vars inited')
    print(tfsession.run(M.Keys))
    print(tfsession.run(M.Values))
    return tfsession.run(yhats, feed_dict={x:x_})


    


def training_step(session, optimizer, batch_features, batch_labels):
    _, h = session.run([optimizer, embeddings],
                feed_dict={
                    x: batch_features,
                    y: batch_labels
                })
    return h

def print_stats(session, train_features, train_labels, valid_features, valid_labels):
    valid_acc, valid_cost = session.run([accuracy, cost],
                         feed_dict={
                             x: valid_features,
                             y: valid_labels
                         })
    train_acc, train_cost = session.run([accuracy, cost],
                         feed_dict={
                             x: train_features,
                             y: train_labels
                         })
    print('Accuracy: {:.6f} \t Cost: {:.6f}'.format(train_acc, train_cost), '\t' + 'Validation Accuracy: {:.6f} \t Cost: {:.6f}'.format(valid_acc, valid_cost))


# Initializing the variables
sess.run(tf.global_variables_initializer())

print('before training: ', tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SECOND_STAGE'))
print('before training: ', tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

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
            
    print('Epoch {:>2}:\t'.format(epoch + 1), end='')
    #print_stats(sess, x_val, y_val)
    print_stats(sess, x_train, y_train, x_val, y_val)





print(M.Keys)
print('before predicting: ', sess.run(M.Keys))

yhats = predict(x_val, sess)

print(M.Keys)
print('after predicting: ', sess.run(M.Keys))


# print('HEREEEEE1')
# print(np.array(yhats))
# print(np.array(yhats).shape)
# print('HEREEEEE2')
# print(y)
# print('HEREEEEE3')
correct_pred = tf.equal(tf.argmax(np.squeeze(np.array(yhats), axis=1), 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

print(sess.run(accuracy, feed_dict={y: y_val}))


#sess.close()



