import pickle
import numpy as np
from keras.datasets import cifar10
from keras.utils import to_categorical
import sys
import os

from CNN_VK import CNN_VK
from fit_evaluate import fit_evaluate
from data_loader import get_dataset, sample_data

# Hyperparameters:
batch_size = 64
epochs = 500
embedding_dim = 100
learning_rate = 0.001

# Hyperparameters:
hp_dict = {
    'bandwidth': float(sys.argv[1]),
    'n_keys_per_class': int(sys.argv[2]),
    'train_percentage': float(sys.argv[3]),
}

bandwidth = hp_dict['bandwidth']
n_keys_per_class = hp_dict['n_keys_per_class']
train_percentage = hp_dict['train_percentage']

# get training data
x_train, x_val, x_test, y_train, y_val, y_test = get_dataset('cifar10', normalize=True, ratio=train_percentage)
num_categories = y_train.shape[1]
N,h,w,c = x_train.shape
input_shape=h,w,c

model = CNN_VK(
    num_categories,
    input_shape=input_shape, 
    layers=[32, 64, 512], 
    embedding_dim=embedding_dim, 
    n_keys_per_class=n_keys_per_class, 
    bandwidth=bandwidth)

modelpath = F"CNNVK_bw={hp_dict['bandwidth']}_kpc={hp_dict['n_keys_per_class']}_tp={hp_dict['train_percentage']}"

metrics_dict, scores = fit_evaluate(model, x_train, y_train, x_val, y_val, x_test, y_test, batch_size, epochs, lr=learning_rate, logstring=F'tb_logs/{modelpath}')

out_results = (hp_dict, metrics_dict, scores)
filename = F"gridresults/{modelpath}.pkl"
with open(filename, 'wb') as f:
  pickle.dump(out_results, f)
