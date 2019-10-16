import os
import sys
from CNN import CNN
from data_loader import get_dataset
from batch import batch
import tensorflow as tf
import numpy as np
import keras
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
epochs = 1
learning_rate = 0.001
embedding_dim = 100

print('Hyperparameters are set!')

model = CNN(
    num_categories,
    input_shape=input_shape, 
    layers=[32, 64, 512], 
    embedding_dim=embedding_dim)

model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                metrics=['accuracy'])

tbCallBack = keras.callbacks.TensorBoard(log_dir='tb_logs/mainmodel', histogram_freq=0,  
        write_graph=True, write_images=True)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
    min_delta=0, patience=10, verbose=2, mode='auto', restore_best_weights=True)

history = model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_val, y_val), 
        callbacks = [tbCallBack, early_stopping])


train_acc = history.history['acc']
val_acc = history.history['val_acc']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

cfg = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
tg = [10, 20, 30, 40, 50, 60, 70, 80, 90]
lg = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
metrics = ['train_acc', 'val_acc', 'train_loss', 'val_loss', 'mem_acc_val', 'comb_acc_val']

metrics_dict = {}
for cf in cfg:
    for t in tg:
        for l in lg:
            metrics_dict[(cf, t, l)] = {}
            for m in metrics:
                metrics_dict[(cf, t, l)][m] = []

best_HP = ()
best_acc = 0
for cf in cfg:
    for t in tg:
        for l in lg:
            try:
                mem_acc_val, comb_acc_val = memory_predictions(model, x_train, x_val, y_train, y_val, num_categories, cf, t, l)
            except:
                mem_acc_val, comb_acc_val= (0, 0)
            for m in metrics:
                metrics_dict[(cf, t, l)][m].append(eval(m))
            if metrics_dict[(cf, t, l)]['comb_acc_val'][-1] > best_acc:
                best_acc = metrics_dict[(cf, t, l)]['comb_acc_val']
                best_HP = (cf, t, l)

with open('metrics_dict.pickle', 'wb') as f:
    pickle.dump(metrics_dict, f)

bcf, bt, bl = best_HP
# add 'mem_acc_train', 'comb_acc_train' to metrics
metrics = ['train_acc', 'val_acc', 'test_acc', 'train_loss', 'val_loss', 'test_loss', 
            'mem_acc_train', 'comb_acc_train', 'mem_acc_val', 'comb_acc_val', 
            'mem_acc_train_val', 'comb_acc_train_val', 'mem_acc_test', 'comb_acc_test', ]
best_metrics = {}
for m in metrics:
    best_metrics[m] = []

# set the max number of epochs equal to the pure CNN learning
epochs = len(train_acc)

# Testing the optimal hyperparameters on the test set:
x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')
y = tf.placeholder(tf.float32, shape=(None, 10), name='output_y')
pred = model(x)
cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=pred, onehot_labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

x_train_val = np.vstack((x_train, x_val))
y_train_val = np.vstack((y_train, y_val))

# training model with optimal hyperparameters
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    
    for epoch in range(epochs):

        idx = np.random.randint(0, x_train_val.shape[0], x_train_val.shape[0])
        num_batches = int(x_train_val.shape[0]/batch_size)
        batch_idxs = np.array_split(idx, num_batches)

        for bidx in batch_idxs:
            # train
            sess.run(optimizer, feed_dict = {x: x_train_val[bidx], y: x_train_val[bidx]})
        print(F'epoch: {epoch}/{epochs}')

        # get accuracy
        train_acc = sess.run(accuracy, feed_dict = {x: x_train[:10000], y: y_train[:10000]})
        val_acc = sess.run(accuracy, feed_dict = {x: x_val, y: y_val})
        test_acc = sess.run(accuracy, feed_dict = {x: x_test, y: y_test})

        # get loss
        train_loss = sess.run(cost, feed_dict = {x: x_train[:10000], y: y_train[:10000]})
        val_loss = sess.run(cost, feed_dict = {x: x_val, y: y_val})
        test_loss = sess.run(cost, feed_dict = {x: x_test, y: y_test})

        # wrapped in try-except due to Overflow/Memory errors
        try:
            mem_acc_train, comb_acc_train = memory_predictions(model, x_train, x_train, y_train, y_train, num_categories, bcf, bt, bl)
        except Exception as e:
            print(e)
            mem_acc_train, comb_acc_train = (e, e)

        try:
            mem_acc_val, comb_acc_val = memory_predictions(model, x_train, x_val, y_train, y_val, num_categories, bcf, bt, bl)
        except Exception as e:
            print(e)
            mem_acc_val, comb_acc_val = (e, e)
        
        try:
            mem_acc_train_val, comb_acc_train_val = memory_predictions(model, x_train, x_train_val, y_train, y_train_val, num_categories, bcf, bt, bl)
        except Exception as e:
            print(e)
            mem_acc_train_val, comb_acc_train_val = (e, e)

        try:
            mem_acc_test, comb_acc_test = memory_predictions(model, x_train, x_test, y_train, y_test, num_categories, bcf, bt, bl)
        except Exception as e:
            print(e)
            mem_acc_train, comb_acc_train = (e, e)

        for m in metrics:
            best_metrics[m].append(eval(m))
        
        with open('best_metrics.pickle', 'wb') as f:
            pickle.dump(best_metrics, f)

        print('Memory Metrics recorded and all metrics saved.')
