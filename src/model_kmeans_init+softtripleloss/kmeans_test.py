import tensorflow as tf 
import numpy as np 
from random_mini_batches import random_mini_batches
from data_loader import get_dataset
import sys 

from model import conv_netV2





def sample(x, y, tr, n_classes=10):

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


    return (sample_x, sample_y)
 
num_classes = 10
x_placeholder = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')
y_placeholder = tf.placeholder(tf.float32, shape=(None, num_classes), name='output_y')
encoder = conv_netV2(x_placeholder, 10)


num_iterations = 5 
keys_per_class = 3
split_ratio = 0.2
train_ratio = 0.1
dataset = 'cifar10'

x_train, x_val, x_test, y_train, y_val, y_test = get_dataset(dataset, ratio=split_ratio, normalize=True)

x, y = sample(x_train, y_train, train_ratio)
x = x[0] # take first class
y = y[0] # take first class


batches = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size=32, drop_remainder=True)   
with tf.Session() as sess:
    out = sess.run([encoder], feed_dict={x_placeholder : x})
    print(out)



num_clusters = keys_per_class

kmeans = tf.contrib.factorization.KMeansClustering(num_clusters=num_clusters, use_mini_batch=False,
                                            initial_clusters=tf.contrib.factorization.KMeansClustering.KMEANS_PLUS_PLUS_INIT)

input_fn =  lambda : tf.data.Dataset.from_tensors([x, y]).repeat(1)      
# train
previous_centers = None
for _ in range(num_iterations):
    kmeans.train(input_fn)
    cluster_centers = kmeans.cluster_centers()
    if previous_centers is not None:
        print( 'delta:', cluster_centers - previous_centers )
    previous_centers = cluster_centers
    print( 'score:', kmeans.score(input_fn))
    print( 'cluster centers:', cluster_centers)