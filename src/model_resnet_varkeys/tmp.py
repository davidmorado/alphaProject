


from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow.keras as keras
from tensorflow.keras.layers import Layer

from tensorflow.keras import layers
from tensorflow.keras import optimizers
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

import sys

from model import conv_netV2
from varkeys import Varkeys
from random_mini_batches import random_mini_batches
from data_loader import get_dataset
from tensorflow.keras.layers import Input, Flatten, BatchNormalization, Dense, Dropout, UpSampling2D



# hyperparameters
epochs = 100
batch_size = 32
learning_rate = 0.001
embedding_size = 100
dataset = 'cifar10'
split_ratio = 0.2
n_output = num_classes= 10
tr = 1
keys_per_class = 2
kmeans_max_iter = 100
reg = 0.1


import os
# creates folders
folders = ['models', 'gridresults', 'evaluation', 'evaluation/cifar10', 'evaluation/cifar100', 'plots', 'tb_logs', 'errs', 'logs']
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




# Data Loading and Preprocessing
x_train, x_val, x_test, y_train, y_val, y_test = get_dataset(dataset, ratio=split_ratio, normalize=True)



x_train = x_train[:5000].astype(np.float32)
y_train = y_train[:5000].astype(np.float32)
x_val = x_val[:1000].astype(np.float32)
y_val = y_val[:1000].astype(np.float32)
x_test = x_test[:1000].astype(np.float32)
y_test = y_test[:1000].astype(np.float32)





def sample(x, y, tr, n_classes):

    sample_x = []
    sample_y = []
    
    for category in range(n_classes):
        idx_category = [idx for idx in range(y.shape[0]) if  y[idx, category] == 1]
        x_tmp = x[idx_category] # all members from one class
        y_tmp = y[idx_category]
        n = int(x_tmp.shape[0] * tr)
        np.random.seed(1)
        index = np.random.choice(x_tmp.shape[0], n, replace=False)  

        sample_x.append(x_tmp[index])
        sample_y.append(y_tmp[index])
        # todo: make sampling random instead of taking first n

    # concatenate all classes to one dataset
    sample_x = np.concatenate(sample_x, axis=0)
    sample_y = np.concatenate(sample_y, axis=0)

    # finally shuffle
    np.random.seed(1)
    shuffled_index = np.random.choice(sample_x.shape[0], sample_x.shape[0], replace=False) 
    return (sample_x[shuffled_index], sample_y[shuffled_index])
 


def training_step(session, optimizer, batch_features, batch_labels):
    _, h = session.run([optimizer, embeddings], feed_dict={x : batch_features, y : batch_labels, M.get_bs_placeholder() : batch_features.shape[0]})
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
    plt.clf()

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



def build_graph(sess):
    

    # Inputs
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')
    y = tf.placeholder(tf.float32, shape=(None, num_classes), name='output_y')
    training_flag = tf.placeholder_with_default(False, shape=())

    # network 
    upsampled_image = UpSampling2D((2,2))(x)
    upsampled_image = UpSampling2D((2,2))(upsampled_image)
    upsampled_image = UpSampling2D((2,2))(upsampled_image)
    conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    embeddings = conv_base(upsampled_image)
    embeddings = Flatten()(embeddings)
    embeddings = BatchNormalization()(embeddings)
    embeddings = Dense(128, activation='relu')(embeddings)
    embeddings = Dropout(0.5)(embeddings, training=training_flag)
    embeddings = BatchNormalization()(embeddings)
    embeddings = Dense(embedding_size, activation='relu')(embeddings)

    # memory
    M = Varkeys(sess=sess, encoder=embeddings, x_placeholder=x, keysize=embedding_size, keys_per_class=keys_per_class, 
                num_categories=num_classes, bandwidth=0.1, kmeans_max_iter=kmeans_max_iter)
    output = M(embeddings)
    

    # Loss and Optimizer
    #cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=y))
    cost = tf.reduce_mean(keras.losses.categorical_crossentropy(y, output)) + reg * M.regularizer()

    original_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #optimizer = tf.contrib.estimator.clip_gradients_by_norm(original_optimizer, clip_norm=2.0)
    train_op  = original_optimizer.minimize(cost)

    # Accuracy
    correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
    
    return x, y, training_flag, embeddings, M, output, cost, original_optimizer, train_op, correct_pred, accuracy


def del_graph():
    del x, y, training_flag, embeddings, M, output, cost, original_optimizer, train_op, correct_pred, accuracy






# run best solution on different training ratios 

x_train = np.concatenate([x_train, x_val], axis=0)
y_train = np.concatenate([y_train, y_val], axis=0)
x_val = x_test
y_val = y_test





# for tr in [0.1, 0.2, 0.3, 0.5, 0.75, 1]:
for tr in [1]:
    hp_dict = {'keys_per_class' : 3, 'tr' : tr} 
    history = create_empty_history()

    tf.reset_default_graph()        
    with K.get_session() as sess:

        # build graph
        x, y, training_flag, embeddings, M, output, cost, original_optimizer, train_op, correct_pred, accuracy = build_graph(sess)
        x_train_sample, y_train_sample = sample(x_train, y_train, tr, num_classes)
        

        # Initializing the variables and fix the graph
        sess.run(tf.global_variables_initializer())
        
        data = M.init_keys_random()


    
        # Training cycle
        for epoch in range(epochs):
            
            # Loop over all batches
            minibatches = random_mini_batches(x_train_sample, y_train_sample, batch_size, 1)
            tmp_loss, tmp_acc = [], []
            for i, minibatch in enumerate(minibatches[:-1]):
                batch_X, batch_Y = minibatch
                
                # training step and append to memroy
                # hs = training_step(sess, train_op, batch_X, batch_Y)
                _, hs = sess.run([train_op, embeddings], feed_dict={x : batch_X, y : batch_Y, M.get_bs_placeholder() : batch_X.shape[0], training_flag : True})
                # if i % 100 == 0:
                #     M.update_keys(batch_X, batch_Y) 
                
                # gather loss and accuracy on batch:
                train_acc, train_loss = sess.run([accuracy, cost], feed_dict={x : batch_X, y : batch_Y, M.get_bs_placeholder() : batch_X.shape[0]})
                tmp_acc.append(train_acc)
                tmp_loss.append(train_loss)

            

            # compute training accuracy and loss
            train_acc = np.mean(tmp_acc)
            train_loss = np.mean(tmp_loss)
            history['train']['no_memory']['acc'].append( (epoch, train_acc) )
            history['train']['no_memory']['loss'].append( (epoch, train_loss) )

            # compute validation accuracy and loss
            valid_acc, valid_loss = sess.run([accuracy, cost], feed_dict={x : x_val, y : y_val, M.get_bs_placeholder() : x_val.shape[0]})
            history['valid']['no_memory']['acc'].append( (epoch, valid_acc) )
            history['valid']['no_memory']['loss'].append( (epoch, valid_loss) )


            print('Epoch {:>2}:\t'.format(epoch + 1), end='')
            print('acc: {:.4f}, loss: {:.4f}'.format(train_acc, train_loss), '\t' + 'val_acc: {:.4f}, loss: {:.4f}'.format(valid_acc, valid_loss))


        modelpath = 'BEST_' + '&'.join([F"{param}={value}" for param, value in hp_dict.items()])
        save_history(history=history, hp_dict=hp_dict, modelpath=modelpath, gridsearch=False)
        
        M.keys_heatmap()
        M.plot_keys()
