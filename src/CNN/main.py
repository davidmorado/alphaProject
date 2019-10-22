import tensorflow as tf

from model import conv_netV2


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
dataset = 'cifar10'
split_ratio = 0.15
n_output = num_classes= 10
nodes_in_extra_layer = 50
dropout_in_extra_layer = 0.25
tr = 1



import os
# creates folders
folders = ['models', 'gridresults', 'evaluation', 'plots', 'tb_logs', 'errs', 'logs']
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



# Data Loading and Preprocessing
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_val, x_test, y_train, y_val, y_test = get_dataset(dataset, ratio=split_ratio, normalize=True)



x_train = x_train[:500].astype(np.float32)
y_train = y_train[:500].astype(np.float32)
x_val = x_val[:100].astype(np.float32)
y_val = y_val[:100].astype(np.float32)




# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')
y = tf.placeholder(tf.float32, shape=(None, num_classes), name='output_y')



def sample(x, y, tr, n_classes):

    sample_x = []
    sample_y = []
    
    for category in range(n_classes):
        idx_category = [idx for idx in range(y.shape[0]) if  y[idx, category] == 1]
        x_tmp = x[idx_category]
        y_tmp = y[idx_category]
        n = int(x_tmp.shape[0] * tr)

        sample_x.append(x_tmp[:n])
        sample_y.append(y_tmp[:n])
        # todo: make sampling random instead of taking first n

    sample_x = np.concatenate(sample_x, axis=0)
    sample_y = np.concatenate(sample_y, axis=0)
    return (sample_x, sample_y)
 


def training_step(session, optimizer, batch_features, batch_labels):
    _, h = session.run([optimizer, embeddings], feed_dict={x : batch_features, y : batch_labels})
    return h


def save_history(history, hp_dict, modelpath, gridsearch=True):        
    out_results = (hp_dict, history)
    if gridsearch:
        filename = F"gridresults/{modelpath}.pkl"
    else:
        filename = F"evaluation/{modelpath}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(out_results, f)

    import matplotlib.pyplot as plt
    plt.plot(history['valid']['no_memory']['acc'])
    plt.plot(history['train']['no_memory']['acc'])
    plt.savefig(F'plots/{modelpath}.png')

def create_empty_history():
    return {
            'train' : {
                        'memory' : {'acc' : [], 'loss':[]},         # lists are of the form: [(epoch, score) ...]            
                        'no_memory' : {'acc' : [], 'loss' : []}
                    }, 
            'valid' : {
                        'memory' : {'acc' : [], 'loss':[]}, 
                        'no_memory' : {'acc' : [], 'loss' : []}
                    }
            }

embeddings = conv_netV2(x, embedding_size=embedding_size)


for nodes_in_extra_layer in [20, 50 ,75]:
    for dropout_in_extra_layer in [0.1, 0.25, 0.5]:

        hp_dict = {'nodes_in_extra_layer' : nodes_in_extra_layer, 'dropout_in_extra_layer' : dropout_in_extra_layer, 'tr' : tr}
        history = create_empty_history()

        # add last layer
        logits = tf.contrib.layers.fully_connected(inputs=embeddings, num_outputs=nodes_in_extra_layer, activation_fn=tf.nn.relu)
        logits = tf.nn.dropout(logits, keep_prob=dropout_in_extra_layer) 

        # Loss and Optimizer
        cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=y))

        original_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        #optimizer = tf.contrib.estimator.clip_gradients_by_norm(original_optimizer, clip_norm=2.0)
        train_op  = original_optimizer.minimize(cost)

        # Accuracy
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')


        with tf.Session() as sess:
            # Initializing the variables
            sess.run(tf.global_variables_initializer())

            x_train_sample, y_train_sample = sample(x_train, y_train, tr, num_classes)

            # Training cycle
            for epoch in range(epochs):
                
                # Loop over all batches
                minibatches = random_mini_batches(x_train_sample, y_train_sample, batch_size, 1)
                tmp_loss, tmp_acc = [], []
                for i, minibatch in enumerate(minibatches):
                    batch_X, batch_Y = minibatch

                    # training step and append to memroy
                    hs = training_step(sess, train_op, batch_X, batch_Y)
                    
                    # gather loss and accuracy on batch:
                    train_acc, train_loss = sess.run([accuracy, cost], feed_dict={x : batch_X, y : batch_Y})
                    tmp_acc.append(train_acc)
                    tmp_loss.append(train_loss)



                # compute training accuracy and loss
                train_acc = np.mean(tmp_acc)
                train_loss = np.mean(tmp_loss)
                history['train']['no_memory']['acc'].append( (epoch, train_acc) )
                history['train']['no_memory']['loss'].append( (epoch, train_loss) )

                # compute validation accuracy and loss
                valid_acc, valid_loss = sess.run([accuracy, cost], feed_dict={x : x_val, y : y_val})
                history['valid']['no_memory']['acc'].append( (epoch, valid_acc) )
                history['valid']['no_memory']['loss'].append( (epoch, valid_loss) )


                print('Epoch {:>2}:\t'.format(epoch + 1), end='')
                print('acc: {:.4f}, loss: {:.4f}'.format(train_acc, train_loss), '\t' + 'val_acc: {:.4f}, loss: {:.4f}'.format(valid_acc, valid_loss), end='')


        modelpath = '&'.join([F"{param}={value}" for param, value in hp_dict.items()])
        save_history(history=history, hp_dict=hp_dict, modelpath=modelpath, gridsearch=True)





# run best solution on different training ratios 

x_train = np.concatenate([x_train, x_val], axis=0)
y_train = np.concatenate([y_train, y_val], axis=0)
x_val = x_test
y_val = y_test


for tr in [0.125, 0.25, 0.5, 1]:
    hp_dict = {'nodes_in_extra_layer' : nodes_in_extra_layer, 'dropout_in_extra_layer' : dropout_in_extra_layer, 'tr' : tr} 
    history = create_empty_history()

        with tf.Session() as sess:
            # Initializing the variables
            sess.run(tf.global_variables_initializer())

            x_train_sample, y_train_sample = sample(x_train, y_train, tr, num_classes)

            # Training cycle
            for epoch in range(epochs):
                
                # Loop over all batches
                minibatches = random_mini_batches(x_train_sample, y_train_sample, batch_size, 1)
                tmp_loss, tmp_acc = [], []
                for i, minibatch in enumerate(minibatches):
                    batch_X, batch_Y = minibatch

                    # training step and append to memroy
                    hs = training_step(sess, train_op, batch_X, batch_Y)
                    
                    # gather loss and accuracy on batch:
                    train_acc, train_loss = sess.run([accuracy, cost], feed_dict={x : batch_X, y : batch_Y})
                    tmp_acc.append(train_acc)
                    tmp_loss.append(train_loss)



                # compute training accuracy and loss
                train_acc = np.mean(tmp_acc)
                train_loss = np.mean(tmp_loss)
                history['train']['no_memory']['acc'].append( (epoch, train_acc) )
                history['train']['no_memory']['loss'].append( (epoch, train_loss) )

                # compute validation accuracy and loss
                valid_acc, valid_loss = sess.run([accuracy, cost], feed_dict={x : x_val, y : y_val})
                history['valid']['no_memory']['acc'].append( (epoch, valid_acc) )
                history['valid']['no_memory']['loss'].append( (epoch, valid_loss) )


                print('Epoch {:>2}:\t'.format(epoch + 1), end='')
                print('acc: {:.4f}, loss: {:.4f}'.format(train_acc, train_loss), '\t' + 'val_acc: {:.4f}, loss: {:.4f}'.format(valid_acc, valid_loss), end='')


        modelpath = 'BEST_' + '&'.join([F"{param}={value}" for param, value in hp_dict.items()])

        save_history(history=history, hp_dict=hp_dict, modelpath=modelpath, gridsearch=False)
