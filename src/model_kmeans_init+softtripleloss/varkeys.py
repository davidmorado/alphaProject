from keras.layers import Layer
from keras.models import Model
from keras.initializers import TruncatedNormal
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
import numpy as np

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
    def __init__(self, keysize, keys_per_class, num_categories, bandwidth):
        self.bandwidth = bandwidth
        self.num_categories = num_categories
        self.keysize = keysize # embedding_dim
        self.keys_per_class = keys_per_class
        self.dict_size = num_categories*n_keykeys_per_classs_per_class
        self.values = tf.constant(np.repeat(np.eye(num_categories, dtype=int), n_keys_per_class, axis = 0), 
            dtype=tf.float32, shape = (self.dict_size, num_categories))

        self.keys = tf.Variable(tf.zeros(shape=(self.dict_size, self.keysize)), name="keys") # kmeans

    def __call__(self, x):
        KV =  tf.matmul(tf.transpose(self.kernel(self.keys, x)), self.values)
        KV_ = tf.diag(tf.reshape(tf.reciprocal(tf.matmul(KV, tf.ones((self.num_categories,1)))), [-1]))
        output = tf.matmul(KV_, KV)
        return output

    def sample(self, x, y, tr):

        samples_by_class_x = []
        samples_by_class_y = []
        
        for category in range(self.num_categories):
            idx_category = [idx for idx in range(y.shape[0]) if  y[idx, category] == 1]
            x_tmp = x[idx_category]
            y_tmp = y[idx_category]
            n = int(x_tmp.shape[0] * tr)

            samples_by_class_x.append(x_tmp[:n])
            samples_by_class_y.append(y_tmp[:n])

        return (samples_by_class_x, samples_by_class_y)

    def train_kmeans(self, x, y):
        # train
        num_iterations = 5
        previous_centers = None
        for _ in xrange(num_iterations):
            kmeans.train(lambda : (x, y))
            cluster_centers = kmeans.cluster_centers()
            if previous_centers is not None:
                print 'delta:', cluster_centers - previous_centers
            previous_centers = cluster_centers
            print 'score:', kmeans.score(input_fn)
            print 'cluster centers:', cluster_centers


    def init_keys(self, x, y, ratio):
        x, y = self.sample(x, y, ratio)

        num_clusters = self.keys_per_class
        for class_x, class_y in zip(x, y):
            kmeans = tf.contrib.factorization.KMeansClustering(num_clusters=num_clusters, use_mini_batch=False)
                    




