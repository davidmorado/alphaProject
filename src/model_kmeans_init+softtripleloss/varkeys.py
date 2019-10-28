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
    def __init__(self, encoder, keysize, keys_per_class, num_categories, bandwidth, num_iterations=5):
        self.encoder = encoder
        self.num_iterations = num_iterations
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
        batches = tf.data.Dataset.from_tensors([x_class, y_class]).repeat(1).batch(batch_size=32)   
        for batch in batches: 
            sess.run([self.encoder], feed_dict={x:x, y:y})

 
    
    def init_keys(self, x, y, data_ratio=0.2)
        num_clusters = keys_per_class
        self.kmeans = tf.contrib.factorization.KMeansClustering(num_clusters=num_clusters, use_mini_batch=True,
                                            initial_clusters=tf.contrib.factorization.KMeansClustering.KMEANS_PLUS_PLUS_INIT)

        x, y = self.sample(x, y, data_ratio)
        x = x[0] # take first class
        y = y[0] # take first class

        for x_class, y_class in zip(x, y):

            input_fn =  lambda : tf.data.Dataset.from_tensors([x_class, y_class]).repeat(1).batch(batch_size=32)   

            # train
            previous_centers = None
            for _ in range(self.num_iterations):
                kmeans.train(input_fn)
                cluster_centers = kmeans.cluster_centers()
                previous_centers = cluster_centers
    



