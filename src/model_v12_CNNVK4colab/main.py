import pickle
import numpy as np
from keras.datasets import cifar10
from keras.utils import to_categorical
import sys
import os

from CNN_VK import CNN_VK
from fit_evaluate import fit_evaluate
from data_loader import get_dataset, sample_data
from utils import assertfolders
assertfolders()


# Hyperparameters:
batch_size = 64
epochs = 500
embedding_dim = 100
learning_rate = 0.001


# bandwith sizes
bws = [1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
# number of keys per class
kpcs = [1, 1e1, 1e2, 1e3]
# percentage of training data used
tps = [0.99, 0.5, 0.25, 0.125, 0.0625]

##########
# TEST MODE
# don't forget to set partition=TEST in template.sh
testing = False
if testing:
	# bandwith sizes
	bws = [10]
	# number of keys per class
	kpcs = [1e3]
	# percentage of training data used
	tps = [0.125]

for bw in bws:
    bw = float(bw)
    for kpc in kpcs:
        kpc = int(kpc)
        for tp in tps:
            tp = float(tp)
            print('bw:{}, kpc:{}, tp:{}'.format(bw, kpc, tp))

            # Hyperparameters:
            hp_dict = {
                'bandwidth': bw,
                'n_keys_per_class': kpc,
                'train_percentage': tp,
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


