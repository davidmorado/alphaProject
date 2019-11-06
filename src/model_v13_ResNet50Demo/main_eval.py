# Use optimal HPs to evaluate the model

import sys
import os
import pickle
import numpy as np
import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
#from keras.applications.resnet50 import ResNet50
from RN50 import ResNet50

from utils import getBatchIndices
from data_loader import get_dataset, percentage_splitter

# Hyperparameters:
batch_size = 64
epochs = 1
embedding_dim = 100
learning_rate = 0.0001

# best hyperparameters:
bw = 100
# kpc = 1
kpc = 10
cifar = 10

# training parameters
tps = [0.1, 0.2, 0.3, 0.5, 0.75, 1.]
tp = 0.1
trainwithkeras = True
trainwithtf = True

# Adam Hyperparameters
b1 = 0.9
b2 = 0.999
e = None

# get training data
x_train, x_val, x_test, y_train, y_val, y_test = get_dataset('cifar{}'.format(cifar), normalize=True, ratio=0.2)
# subsample training data
x_train, y_train = percentage_splitter(x_train, x_val, y_train, y_val, merging=True, random_selection=False, ratio=tp) 
num_categories = y_train.shape[1]
N,h,w,c = x_train.shape
input_shape=h,w,c

if trainwithkeras:

    RN50 = ResNet50(input_shape=(32, 32, 3), include_top=False)
    for layer in RN50.layers[1:-2]:
        layer.trainable = False

    x = RN50.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(embedding_dim, activation='relu')(x)
    predictions = Dense(num_categories, activation='softmax')(x)
    model = Model(inputs=RN50.input, outputs=predictions)

    model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer = keras.optimizers.Adam(
            lr=learning_rate, beta_1=b1, beta_2=b2, epsilon=e),
        metrics=['accuracy'])

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
        min_delta=0, patience=10, verbose=2, mode='auto', restore_best_weights=True)

    tbCallBack = keras.callbacks.TensorBoard(log_dir='tb_logs', histogram_freq=1,
            write_graph=True, write_images=True)

    history = model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(x_val, y_val),
                callbacks = [early_stopping, tbCallBack])

    # delete model since custom layers can not be unpickled
    history.model = None

    acclist = history.history['acc']
    losslist = history.history['loss']

    scores = model.evaluate(x_test, y_test)

    out_results = (acclist, losslist, scores, history)
    with open('RN50keras.pkl', 'wb') as f:
        pickle.dump(out_results, f)

if trainwithtf:

    train_acclist = []
    train_losslist = []

    tf.reset_default_graph()

    RN50 = ResNet50(input_shape=(32, 32, 3), include_top=False)
    for layer in RN50.layers[1:-2]:
        layer.trainable = False

    inputs = tf.placeholder(shape=(None,h,w,c), dtype=tf.float32)
    labels = tf.placeholder(shape=(None,num_categories), dtype=tf.int32)

    x = RN50(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(embedding_dim, activation='relu')(x)
    predictions = Dense(num_categories, activation='softmax')(x)
    loss = keras.losses.categorical_crossentropy(labels, predictions)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=b1, beta2=b2, epsilon=e).minimize(loss)
    accuracy = tf.metrics.accuracy(labels, predictions)

    with tf.Session() as sess:

        for epoch in range(epochs):
            batch_idxs = getBatchIndices(x_train, batch_size)
            for bidx in batch_idxs:
                sess.run(optimizer, feed_dict = {inputs: x_train[bidx], labels: y_train[bidx]})
            train_acclist.append(sess.run(accuracy, feed_dict = {inputs: x_train, labels: y_train}))
            train_losslist.append(sess.run(loss, feed_dict = {inputs: x_train, labels: y_train}))
        
        test_acc = sess.run(accuracy, feed_dict = {inputs: x_test, labels: y_test})
    
    out_results = (train_acclist, train_losslist, test_acc)
    with open('RN50tf.pkl', 'wb') as f:
        pickle.dump(out_results, f)