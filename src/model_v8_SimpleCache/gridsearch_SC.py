import os
import sys
from CNN import CNN
from data_loader import get_dataset
from batch import batch
import tensorflow as tf
import numpy as np
from keras import Model
from memory_predictions import memory_predictions
import pickle

# creates folders
folders = ['models', 'gridresults', 'tb_logs', 'errs', 'logs']
for f in folders:
    try:
        os.makedirs(f)
    except OSError:
        pass

x_train, x_val, x_test, y_train, y_val, y_test = get_dataset('cifar10', normalize=True)
num_categories = y_train.shape[1]
input_shape = x_train.shape[1:]

print('Data has been loaded successfully!')

# Hyperparameters:
batch_size = 64
epochs = 500
learning_rate = 0.001
embedding_dim = 100

print('Hyperparameters are set!')

model = CNN(
    num_categories,
    input_shape=input_shape, 
    layers=[32, 64, 512], 
    embedding_dim=embedding_dim)


# Model graph
x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')
y = tf.placeholder(tf.float32, shape=(None, 10), name='output_y')
pred = model(x)
cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=pred, onehot_labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

print('Graph has been created!')

# cfg = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# tg = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# lg = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
cfg = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
tg = [10, 30, 50, 70, 90]
lg = [0.1, 0.3, 0.5, 0.7, 0.9]
metrics = ['train_acc', 'val_acc', 'test_acc', 'train_loss', 'val_loss', 'test_loss', 
            'mem_acc_val', 'comb_acc_val', 'mem_acc_test', 'comb_acc_test']

metrics_dict = {}
for cf in cfg:
    for t in tg:
        for l in lg:
            metrics_dict[(cf, t, l)] = {}
            for m in metrics:
                metrics_dict[(cf, t, l)][m] = []


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        idx = np.random.randint(0, x_train.shape[0], x_train.shape[0])
        num_batches = int(x_train.shape[0]/batch_size)
        batch_idxs = np.array_split(idx, num_batches)

        for bidx in batch_idxs:

            # train
            sess.run(optimizer, feed_dict = {x: x_train[bidx], y: y_train[bidx]})

            print('Training step taken.')

            # get accuracy
            # train_acc = sess.run(accuracy, feed_dict = {x: x_train, y: y_train})
            train_acc = 'OOM'
            val_acc = sess.run(accuracy, feed_dict = {x: x_val, y: y_val})
            test_acc = sess.run(accuracy, feed_dict = {x: x_test, y: y_test})

            # get loss
            # train_loss = sess.run(cost, feed_dict = {x: x_train, y: y_train})
            train_loss = 'OOM'
            val_loss = sess.run(cost, feed_dict = {x: x_val, y: y_val})
            test_loss = sess.run(cost, feed_dict = {x: x_test, y: y_test})

            print('Standard Metrics recorded.')

            for cf in cfg:
                for t in tg:
                    for l in lg:
                        try:
                            mem_acc_val, comb_acc_val = memory_predictions(model, x_train, x_val, y_train, y_val, num_categories, cf, t, l)
                            mem_acc_test, comb_acc_test = memory_predictions(model, x_train, x_test, y_train, y_test, num_categories, cf, t, l)
                        except:
                            mem_acc_val, comb_acc_val, mem_acc_test, comb_acc_test = ('OF', 'OF', 'OF', 'OF')
                        for m in metrics:
                            metrics_dict[(cf, t, l)][m].append(eval(m))
            
            with open('results2.pickle', 'wb') as f:
                pickle.dump(metrics_dict, f)

            print('Memory Metrics recorded and all metrics saved.')