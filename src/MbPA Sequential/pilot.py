import tensorflow as tf

from model import conv_net, conv_netV2, secondStage, SecondStage
from memory import Memory

from keras.datasets import cifar10
from keras.utils import to_categorical
import numpy as np
import sys

from random_mini_batches import random_mini_batches
from data_loader import get_dataset

import objgraph

# hyperparameters
epochs = 100
batch_size = 32
learning_rate = 0.001
embedding_size = 100
nearest_neighbors = 3
validation_freq = 10
dataset = 'cifar10'
split_ratio = 0.1
n_output = num_classes= 10
memory_size = 1000


import os
# creates folders
folders = ['models', 'gridresults', 'plots', 'tb_logs', 'errs', 'logs']
for f in folders:
    try:
        os.makedirs(f)
    except OSError:
        pass


# read hyperparameters from command line arguments and overwrite default ones
# hp_dict_str = sys.argv[1]
# import yaml
# hp_dict = yaml.load(hp_dict_str)

# #hp_dict = ast.literal_eval(hp_dict_str)
# for key,val in hp_dict.items():
#     exec(key + '=val')
# print('nearest_neighbors: ',nearest_neighbors)
# print('update_period: ',update_period)


# lists for storing data about accuracy and loss:
history = {
    'train' : {
                'memory' : {'acc' : [], 'loss':[]},         # lists are of the form: [(epoch, score) ...]            
                'no_memory' : {'acc' : [], 'loss' : []}
            }, 
    'valid' : {
                'memory' : {'acc' : [], 'loss':[]}, 
                'no_memory' : {'acc' : [], 'loss' : []}
            }
}


# Data Loading and Preprocessing
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_val, x_test, y_train, y_val, y_test = get_dataset(dataset, ratio=split_ratio, normalize=True)

x_train = np.concatenate([x_train, x_val], axis=0)
y_train = np.concatenate([y_train, y_val], axis=0)
x_val = x_test
y_val = y_test


# x_train = x_train/255
# x_val = x_val/255
# y_train = to_categorical(y_train, num_classes)
# y_val = to_categorical(y_val, num_classes)

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
y = tf.placeholder(tf.float32, shape=(None, num_classes), name='output_y')


# start tf session
sess = tf.Session()

# build network with memory
M = Memory(embedding_size=embedding_size, size=memory_size, session=sess, target_size=num_classes, K=nearest_neighbors)
            
M.initialize()
embeddings = conv_netV2(x, embedding_size=embedding_size)
logits = M.model(embeddings)

# Loss and Optimizer
cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=y))

original_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#optimizer = tf.contrib.estimator.clip_gradients_by_norm(original_optimizer, clip_norm=2.0)
train_op  = original_optimizer.minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

# accuracy when memroy is used
yhat_placeholder = tf.placeholder(tf.float32, shape=(None, num_classes), name='yhat_placeholder')
correct_pred_mem = tf.equal(tf.argmax(yhat_placeholder, 1), tf.argmax(y, 1))
accuracy_mem = tf.reduce_mean(tf.cast(correct_pred_mem, tf.float32), name='accuracy_memory')

# print(sess.run(accuracy, feed_dict={y: y_val}))


def predict(x_, tfsession):
    # x_: [batchsize x 32 x 32 x 3]

    hs_ = tfsession.run(embeddings, feed_dict={x:x_})
    yhats = M.predict(hs_)
    r =  tfsession.run(yhats, feed_dict={x:x_})
    return np.squeeze(np.array(r), axis=1)

def training_step(session, optimizer, batch_features, batch_labels):
    _, h = session.run([optimizer, embeddings], feed_dict={x : batch_features, y : batch_labels})
    return h


# Initializing the variables
sess.run(tf.global_variables_initializer())


# Training cycle
for epoch in range(epochs):
    
    # Loop over all batches
    minibatches = random_mini_batches(x_train, y_train, batch_size, 1)
    # minibatches = minibatches[:-1]
    tmp_loss, tmp_acc = [], []
    hs_, y_ = [], []
    for i, minibatch in enumerate(minibatches):
        batch_X, batch_Y = minibatch

        # training step and append to memroy
        hs = training_step(sess, train_op, batch_X, batch_Y)
        if i == 0:
            hs_ = hs
            y_ = batch_Y
        else:
            hs_ = np.concatenate([hs_, hs], axis=0)
            y_ = np.concatenate([y_, batch_Y], axis=0)
        
        # gather loss and accuracy on batch:
        train_acc, train_loss = sess.run([accuracy, cost], feed_dict={x : batch_X, y : batch_Y})
        tmp_acc.append(train_acc)
        tmp_loss.append(train_loss)

    # update memory
    M.write(hs_, y_) 

    # compute training accuracy and loss
    train_acc = np.mean(tmp_acc)
    train_loss = np.mean(tmp_loss)
    history['train']['no_memory']['acc'].append( (epoch, train_acc) )
    history['train']['no_memory']['loss'].append( (epoch, train_loss) )

    # compute validation accuracy and loss
    valid_acc, valid_loss = sess.run([accuracy, cost], feed_dict={x : x_val, y : y_val})
    history['valid']['no_memory']['acc'].append( (epoch, valid_acc) )
    history['valid']['no_memory']['loss'].append( (epoch, valid_loss) )

    # compute validation accuracy when using memory
    if(epoch+1) % validation_freq == 0 or epoch == epochs-1:
        # print(M.Keys)
        # print('before predicting: ', sess.run(M.Keys))
        yhats = predict(x_val, sess)
        # print(M.Keys)
        # print('after predicting: ', sess.run(M.Keys))
        mem_val_acc = sess.run(accuracy_mem, feed_dict={y: y_val, yhat_placeholder : yhats})
        history['valid']['memory']['acc'].append( (epoch, mem_val_acc) )

    print('Epoch {:>2}:\t'.format(epoch + 1), end='')
    print('acc: {:.4f}, loss: {:.4f}'.format(train_acc, train_loss), '\t' + 'val_acc: {:.4f}, loss: {:.4f}'.format(valid_acc, valid_loss), end='')
    if (epoch+1) % validation_freq == 0 or epoch == epochs-1:
        print('\t memory val_acc: {:.4f}'.format(mem_val_acc)) 
    else:
        print()






print(M.Keys)
print('before predicting: ', sess.run(M.Keys))

yhats = predict(x_val, sess)

print(M.Keys)
print('after predicting: ', sess.run(M.Keys))

correct_pred = tf.equal(tf.argmax(np.squeeze(np.array(yhats), axis=1), 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy_memory')
print(sess.run(accuracy, feed_dict={y: y_val}))


modelpath = 'mbpa'

out_results = history
filename = F"gridresults/{modelpath}.pkl"
with open(filename, 'wb') as f:
  pickle.dump(out_results, f)


# close tf session
sess.close()



