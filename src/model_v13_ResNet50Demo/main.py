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
from RN50 import ResNet50, cResNet50

from utils import getBatchIndices
from data_loader import get_dataset, percentage_splitter

# Hyperparameters:
batch_size = 64
epochs = 150
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
trainwithkeras = False
trainwithtf = True

# Adam Hyperparameters
b1 = 0.9
b2 = 0.999
e = 0.1

# get training data
x_train, x_val, x_test, y_train, y_val, y_test = get_dataset('cifar{}'.format(cifar), normalize=True, ratio=0.2)
# subsample training data
x_train, y_train = percentage_splitter(x_train, x_val, y_train, y_val, merging=True, random_selection=False, ratio=tp)
num_categories = y_train.shape[1]
N,h,w,c = x_train.shape
input_shape=h,w,c



if trainwithtf:

    train_acclist = []
    train_losslist = []

    inputs = tf.placeholder(shape=(None,h,w,c), dtype=tf.float32)
    labels = tf.placeholder(shape=(None,num_categories), dtype=tf.float32)

    x = cResNet50(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(embedding_dim, activation='relu')(x)
    predictions = Dense(num_categories, activation='softmax')(x)
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=predictions, onehot_labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=b1, beta2=b2, epsilon=e).minimize(loss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1)), tf.float32))

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        tf.summary.FileWriter('tmp/graphs', sess.graph)
        print('Graph written!')

        for epoch in range(epochs):
            batch_idxs = getBatchIndices(x_train, batch_size)
            for bidx in batch_idxs:
                sess.run(optimizer, feed_dict = {inputs: x_train[bidx], labels: y_train[bidx]})
            acc = sess.run(accuracy, feed_dict = {inputs: x_train, labels: y_train})
            losss = sess.run(loss, feed_dict = {inputs: x_train, labels: y_train})
            train_acclist.append(acc)
            train_losslist.append(losss)
            print(acc)
            print(losss)
        
        test_acc = sess.run(accuracy, feed_dict = {inputs: x_test, labels: y_test})
    
    out_results = (train_acclist, train_losslist, test_acc)
    with open('RN50tf.pkl', 'wb') as f:
        pickle.dump(out_results, f)


























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

    print('Keras model compiled!')

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
        min_delta=0, patience=10, verbose=2, mode='auto', restore_best_weights=True)

    tbCallBack = keras.callbacks.TensorBoard(log_dir='tb_logs', histogram_freq=1,
            write_graph=True, write_images=True)

    history = model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=2,
                validation_data=(x_val, y_val),
                callbacks = [early_stopping])#, tbCallBack])

    print('Keras model fitted to data!')

    # delete model since custom layers can not be unpickled
    history.model = None

    acclist = history.history['acc']
    losslist = history.history['loss']

    scores = model.evaluate(x_test, y_test)

    print('Keras model evaluated!')

    out_results = (acclist, losslist, scores, history)
    with open('RN50keras.pkl', 'wb') as f:
        pickle.dump(out_results, f)

# problems
# Tensorflow: 
# /content/drive/My Drive/model_v13_ResNet50Demo/RN50.py:266: UserWarning: The output shape of `ResNet50(include_top=False)` has been changed since Keras 2.2.0.
#   warnings.warn('The output shape of `ResNet50(include_top=False)` '
# Traceback (most recent call last):
#   File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/client/session.py", line 1120, in _run
#     subfeed, allow_tensor=True, allow_operation=False)
#   File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/framework/ops.py", line 3607, in as_graph_element
#     return self._as_graph_element_locked(obj, allow_tensor, allow_operation)
#   File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/framework/ops.py", line 3686, in _as_graph_element_locked
#     raise ValueError("Tensor %s is not an element of this graph." % obj)
# ValueError: Tensor Tensor("Placeholder:0", shape=(7, 7, 3, 64), dtype=float32) is not an element of this graph.

# During handling of the above exception, another exception occurred:

# Traceback (most recent call last):
#   File "main_eval.py", line 107, in <module>
#     RN50 = ResNet50(input_shape=(32, 32, 3), include_top=False)
#   File "/content/drive/My Drive/model_v13_ResNet50Demo/RN50.py", line 292, in ResNet50
#     model.load_weights(weights_path)
#   File "/usr/local/lib/python3.6/dist-packages/keras/engine/saving.py", line 458, in load_wrapper
#     return load_function(*args, **kwargs)
#   File "/usr/local/lib/python3.6/dist-packages/keras/engine/network.py", line 1217, in load_weights
#     f, self.layers, reshape=reshape)
#   File "/usr/local/lib/python3.6/dist-packages/keras/engine/saving.py", line 1199, in load_weights_from_hdf5_group
#     K.batch_set_value(weight_value_tuples)
#   File "/usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py", line 2732, in batch_set_value
#     get_session().run(assign_ops, feed_dict=feed_dict)
#   File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/client/session.py", line 956, in run
#     run_metadata_ptr)
#   File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/client/session.py", line 1123, in _run
#     e.args[0])
# TypeError: Cannot interpret feed_dict key as Tensor: Tensor Tensor("Placeholder:0", shape=(7, 7, 3, 64), dtype=float32) is not an element of this graph.