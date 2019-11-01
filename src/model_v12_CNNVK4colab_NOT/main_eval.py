# Use optimal HPs to evaluate the model

import pickle
import numpy as np
import keras
from keras.datasets import cifar10
from keras.utils import to_categorical
import sys
import os

from CNN_VK import CNN_VK
from data_loader import get_dataset, percentage_splitter
from utils import assertfolders
assertfolders()

# Hyperparameters:
batch_size = 64
epochs = 500
embedding_dim = 100
learning_rate = 0.001

# best hyperparameters:
bw = 100
kpc = 1
tp = 1.

# get training data
x_train, x_val, x_test, y_train, y_val, y_test = get_dataset('cifar10', normalize=True, ratio=0.2)
# subsample training data
x_train, y_train = percentage_splitter(x_train, x_val, y_train, y_val, merging=True, random_selection=False, ratio=tp) 
num_categories = y_train.shape[1]
N,h,w,c = x_train.shape
input_shape=h,w,c

model = CNN_VK(
    num_categories,
    input_shape=input_shape, 
    layers=[32, 64, 512], 
    embedding_dim=embedding_dim, 
    n_keys_per_class=kpc, 
    bandwidth=bw)

model.compile(loss=keras.losses.categorical_crossentropy,
    optimizer = keras.optimizers.Adam(
        lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
    metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(monitor='loss',
    min_delta=0, patience=10, verbose=2, mode='auto', restore_best_weights=True)

history = model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        callbacks = [early_stopping])

# memory = model.get_weights()[-1]

acc = history.history['acc']
loss = history.history['loss']

scores = model.evaluate(x_test, y_test)

out_results = (acc, loss, scores)
filename = F"finalresults.pkl"
with open(filename, 'wb') as f:
    pickle.dump(out_results, f)


