from keras.layers import Layer
from keras.models import Model
from keras.initializers import TruncatedNormal
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
import numpy as np
from model import conv_netV2
from random_mini_batches import random_mini_batches
from data_loader import get_dataset
from sklearn.cluster import KMeans

# class Varkeys(Layer):

#     def __init__(self, keysize, n_keys_per_class, num_categories, bandwidth, **kwargs):
#         self.bandwidth = bandwidth
#         self.initializer = TruncatedNormal(mean=0.0, stddev=3, seed=None)
#         self.num_categories = num_categories
#         self.keysize = keysize # embedding_dim
#         self.dict_size = num_categories*n_keys_per_class
#         self.values = tf.constant(np.repeat(np.eye(num_categories, dtype=int), n_keys_per_class, axis = 0), 
#             dtype=tf.float32, shape = (self.dict_size, num_categories))
#         super(Varkeys, self).__init__(**kwargs)

#     def build(self, input_shape):
#         # Create a trainable weight variable for this layer.
#         self.keys = self.add_weight(name='keys', 
#                                       shape=(self.dict_size, self.keysize),
#                                       initializer=self.initializer,
#                                       trainable=True)
#         super(Varkeys, self).build(input_shape)

#     def call(self, x):
#         # switch out with KNN if you want!
#         KV =  tf.matmul(tf.transpose(self.kernel(self.keys, x)), self.values)
#         KV_ = tf.diag(tf.reshape(tf.reciprocal(tf.matmul(KV, tf.ones((self.num_categories,1)))), [-1]))
#         output = tf.matmul(KV_, KV)
#         return output

    
# def sq_distance(self, A, B):
#     row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
#     row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

#     row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
#     row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

#     return row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B


# def kernel (self, A,B):
#     d = self.sq_distance(A,B)/self.bandwidth
#     o = tf.reciprocal(d+1e-4)
#     return o



class Varkeys:
    def __init__(self, sess, encoder, x_placeholder, keysize, keys_per_class, num_categories, bandwidth, num_iterations_kmeans=5):
        self.encoder = encoder
        self.encoder_placeholder = x_placeholder
        self.sess = sess
        self.num_iterations = num_iterations_kmeans
        self.bandwidth = bandwidth
        self.num_categories = num_categories
        self.keysize = keysize # embedding_dim
        self.keys_per_class = keys_per_class
        self.dict_size = num_categories*keys_per_class
        self.initialized = False
        values = [[self.one_hot(self.num_categories, i)] * self.keys_per_class for i in range(self.num_categories)] 
        self.values = tf.constant(values, dtype=tf.float32, shape = (self.dict_size, self.num_categories))

        self.keys_init_placeholder = tf.placeholder(tf.float32, shape=(self.dict_size, self.keysize))
        self.keys = tf.get_variable(name="key", initializer=tf.zeros_like(self.keys_init_placeholder))
        self.keys_init = tf.assign(self.keys, self.keys_init_placeholder)

    def __call__(self, x):
        KV =  tf.matmul(tf.transpose(self.kernel(self.keys, x)), self.values)
        KV_ = tf.diag(tf.reshape(tf.reciprocal(tf.matmul(KV, tf.ones((self.num_categories,1)))), [-1]))
        output = tf.matmul(KV_, KV)
        return output

    def sq_distance(self, A, B):
        row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
        row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

        row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
        row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

        return row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B


    def kernel (self, A,B):
        d = self.sq_distance(A,B)/self.bandwidth
        o = tf.reciprocal(d+1e-4)
        return o



    def sample(self, x, y, tr, n_classes=10):

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


    def _input_fn(self, x, y):         
        embeddings = []
        labels = []
        batch_size=32
        minibatches = random_mini_batches(x, y, batch_size, 1)
        for minibatch in minibatches:
            batch_x, batch_y = minibatch
            out = self.sess.run(self.encoder, feed_dict={self.encoder_placeholder : batch_x})
            embeddings.append(out)
            labels.append(batch_y)

            
        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)

        def input_fn():
            batches = tf.data.Dataset.from_tensor_slices((embeddings)).batch(batch_size=64, drop_remainder=True) 
            return batches 

        return input_fn, embeddings, labels


    def one_hot(self, length, i):
        return [1 if idx==i else 0 for idx in range(length)]

    
    def init_keys_tf(self, x, y, data_ratio=0.2):
        num_clusters = self.keys_per_class
        self.kmeans = tf.contrib.factorization.KMeansClustering(num_clusters=num_clusters, use_mini_batch=True,
                                            initial_clusters=tf.contrib.factorization.KMeansClustering.KMEANS_PLUS_PLUS_INIT)

        x, y = self.sample(x, y, data_ratio)
        data = [] # for class in classes: (embeddings, labels)

        smart_keys = []
        for x_class, y_class in zip(x, y):

            input_fn, embeddings, labels =  self._input_fn(x_class, y_class)   
            data.append((embeddings, labels))
            # train
            previous_centers = None
            for _ in range(self.num_iterations):
                self.kmeans.train(input_fn)
                cluster_centers = self.kmeans.cluster_centers()
                previous_centers = cluster_centers
                print( 'score:', self.kmeans.score(input_fn))
                print( 'cluster centers:', cluster_centers)

            centers = self.kmeans.cluster_centers()
            smart_keys.append(centers)
        smart_keys = np.concatenate(smart_keys, axis=0)

        del self.kmeans
        
        self.sess.run(self.keys_init, feed_dict={self.keys_init_placeholder:smart_keys})

        self.initialized = True
        return data


    def init_keys(self, x, y, data_ratio=0.2):
        num_clusters = self.keys_per_class
        

        x, y = self.sample(x, y, data_ratio)
        data = [] # for class in classes: (embeddings, labels)

        smart_keys = []
        for x_class, y_class in zip(x, y):

            _, embeddings, labels =  self._input_fn(x_class, y_class)   
            data.append((embeddings, labels))

            kmeans = KMeans(n_clusters=num_clusters, random_state=0, max_iter=100).fit(embeddings)
            centers = kmeans.cluster_centers_
            smart_keys.append(centers)

        smart_keys = np.concatenate(smart_keys, axis=0)
       
        self.sess.run(self.keys_init, feed_dict={self.keys_init_placeholder:smart_keys})

        self.initialized = True
        return data




    

if __name__ == '__main__':
    import matplotlib.pyplot as plt 
    num_classes = 10
    keys_per_class = 3
    dataset = 'cifar10'
    split_ratio = 0.2
    embedding_size = 2
    num_iterations_kmeans = 5
    sess = tf.Session()
    x_placeholder = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')
    y_placeholder = tf.placeholder(tf.float32, shape=(None, num_classes), name='output_y')
    encoder = conv_netV2(x_placeholder, embedding_size)
    m = Varkeys(sess=sess, encoder=encoder, x_placeholder=x_placeholder, keysize=embedding_size, keys_per_class=keys_per_class, num_categories=num_classes, bandwidth=0.1, num_iterations_kmeans=num_iterations_kmeans)
    sess.run(tf.global_variables_initializer())
    x_train, x_val, x_test, y_train, y_val, y_test = get_dataset(dataset, ratio=split_ratio, normalize=True)
    data = m.init_keys(x_train, y_train, data_ratio=0.1)

    
    print(m.keys)
    print(m.values)
    keys = sess.run(m.keys)
    V = sess.run(m.values)
    print(keys)
    print(V)
    colors = ['orange', 'yellow', 'green', 'blue', 'red', 'black', 'violet', 'brown', 'gray', 'lightblue'] 
    for i in range(num_classes):
        x_class, y_class = data[i]
        start_index = i * keys_per_class
        end_index = (i+1) * keys_per_class
        keys_class_i = keys[start_index : end_index]
        plt.plot(keys_class_i[:, 0], keys_class_i[:, 1], 'ro', marker='x', markersize =10, color=colors[i])
        
        #xes
        print(y_class)
        print(y_class[0][i])
        assert y_class[0][i] == 1
        plt.plot(x_class[:, 0], x_class[:, 1], 'ro', marker='o', markersize =5, color=colors[i])
        plt.show()
        plt.clf()
    
    for i in range(num_classes):
        x_class, y_class = data[i]
        start_index = i * keys_per_class
        end_index = (i+1) * keys_per_class
        keys_class_i = keys[start_index : end_index]
        plt.plot(keys_class_i[:, 0], keys_class_i[:, 1], 'ro', marker='x', markersize =10, color=colors[i])
    plt.show()     





