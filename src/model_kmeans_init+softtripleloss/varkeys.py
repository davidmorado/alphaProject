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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix as dist
from seaborn import heatmap 
import sys
from scipy.stats import truncnorm

class KNN:

    def __init__(self, k, bs):
        self.k = k # hyperparameter k for KNN
        self.bs = bs # placeholder for batchsize

    def build(self, x, keys, values):
        # x:  (batch_size x embedding_size)
        # keys (n_keys x embedding_size)
        # values (n_keys x n_classes)
        dict_size = tf.shape(keys)[0]
        keysize = tf.shape(keys)[1]
        print('expected x: ', x)
        
        num_classes = tf.shape(values)[1]
        with tf.variable_scope('KNN'):
            expanded_x = tf.expand_dims(x, axis=1) # shape: (batch_size, 1, embedding size)
            expanded_keys = tf.expand_dims(keys, axis=0) # shape: (1, dict_size, embedding_size)
            tiled_expanded_x = tf.tile(expanded_x, [1, tf.shape(keys)[0], 1]) # shape: (batch_size, dict_size, embedding size)

            # compute distances
            diff = tf.square(expanded_keys - tiled_expanded_x)
            distances = tf.reduce_sum(diff,axis=2)  # shape: (batch_size, dict_size)

            # get nearest neighbors
            _, indices = tf.nn.top_k(-distances, k=self.k)  # shape: [batchsize x k]
            # hit_values = tf.nn.embedding_lookup(values, tf.range(dict_size)) # shape: (batch-size, k, n_classes) 
            # hit_keys = tf.nn.embedding_lookup(keys, tf.range(dict_size)) # (batch-size, k, keysize) 

            keys_expanded = tf.tile(tf.expand_dims(keys, axis=0), [self.bs, 1, 1])     # (batch-size, n_keys, keysize) 
            values_expanded = tf.tile(tf.expand_dims(values, axis=0), [self.bs, 1, 1]) # (batch-size, n_keys, n_classes) 
            print('indices: ', indices, keys_expanded, values_expanded)

            indices2 = indices[:,:,None]
            a = tf.range(self.bs)
            a = tf.reshape(a, [self.bs, 1, 1])
            a = tf.tile(a, [1, self.k, 1])
            indices3 = tf.concat([a, indices2], axis=2)
            updates = tf.ones([self.bs,self.k])
            hit_mask = tf.zeros([self.bs, dict_size], dtype=tf.float32) 
            hit_mask_ = tf.tensor_scatter_nd_update(hit_mask, indices3, updates=updates)

            values_expanded = tf.multiply(tf.tile(tf.expand_dims(hit_mask_, axis=2), [1, 1, num_classes]) , values_expanded) # elementwise multiplication
            return keys_expanded, values_expanded, hit_mask_








class Varkeys:
    def __init__(self, sess, encoder, x_placeholder, keysize, keys_per_class, num_categories, kmeans_max_iter=100,
                    method='standard', knn_k=5, bandwidth=0.1):
        self.encoder = encoder
        self.encoder_placeholder = x_placeholder
        self.sess = sess
        self.kmeans_max_iter = kmeans_max_iter
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

        # intern hyperparameters:
        if method not in ['standard', 'knn']:
            raise Exception(F'Method {method} is not implemented')
        self.call_method = method
        self.k= 5
        self.bs_placeholder = tf.placeholder(tf.int32, shape=[])
        self.knn = KNN(self.k, self.bs_placeholder)
        

    def __call__(self, x):
        if self.call_method=='standard':
            return self.call__standard(x)
        elif self.call_method=='knn':
            return self.call__knn(x)

    def get_bs_placeholder(self):
        return self.bs_placeholder

    def call__standard(self, x):
        print(self.kernel(self.keys, x).shape)
        KV =  tf.matmul(tf.transpose(self.kernel(self.keys, x)), self.values)
        KV_ = tf.diag(tf.reshape(tf.reciprocal(tf.matmul(KV, tf.ones((self.num_categories,1)))), [-1]))
        output = tf.matmul(KV_, KV)
        return output

    def call__knn(self, x):
        
        keys, values, hit_mask_ = self.knn.build(x, self.keys, self.values)
       
        # kernel: [batchsize x k], values: (batchsizee, k, n_classes) 
        # i need: [batchsize x batchsize x k], values: (batchsize, k, n_classes) --> 
        ker = tf.tile(tf.expand_dims(self.kernel_with_batchsize(keys, x), axis=0), [self.bs_placeholder, 1, 1]) # shape [batchsize x batchsize x k]
        KV =  tf.matmul(ker, values) # (k, batchsize, num_classes)
        KV = KV[0,:,:]
        KV_ = tf.diag(tf.reshape(tf.reciprocal(tf.matmul(KV, tf.ones((self.num_categories,1)))), [-1]))
        output = tf.matmul(KV_, KV)
        return output, values, hit_mask_

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


    def sq_distance_with_batchsize(self, A, B):
        # A = hit_keys: [batchsize x K x embedding_size]
        # B = h: [batchsize x embedding_size]
        # computes ||A||^2 - 2*||AB|| + ||B||^2 = A.TA - 2 A.T B + B.T B
        row_norms_A = tf.reduce_sum(tf.square(A), axis=2)
        row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
        row_norms_B = tf.reshape(row_norms_B, [-1, 1])
        # B: [batchsize x embedding_size x 1]
        B = tf.expand_dims(B, axis=2)
        # B: [batchsize x embedding_size x K]
        B = tf.tile(B, [1, 1, self.k])
        # https://stackoverflow.com/questions/38235555/tensorflow-matmul-of-input-matrix-with-batch-data
        # AB = [batchsize x K x embedding_size] @ [batchsize x embedding_size x K] 
        # -> [batchsize x K x K] (duplicated on axis 2)
        AB = tf.matmul(A, B) 
        # AB -> [batchsize x K]
        AB = AB[:,:,0] # last dim is just duplacates
        result = row_norms_A - 2 * AB + row_norms_B
        return result

    def kernel_with_batchsize(self, A,B):
        #   1/(e + tf.square(hit_keys - h))
        distances = self.sq_distance_with_batchsize(A,B)
        weights = tf.reciprocal(distances+tf.constant(1e-4))
        return weights # weight matrix: [batchsize x K]

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
            for _ in range(self.kmeans_max_iter):
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

    def init_keys_kmeans(self, x, y, data_ratio=0.2):
        num_clusters = self.keys_per_class
        x, y = self.sample(x, y, data_ratio)
        data = [] # for class in classes: (embeddings, labels)
        smart_keys = []

        # if tf.report_uninitialized_variables().shape[0] > 0: # this crashes with sess.graph.finalize()
        #     raise Exception('Initialize Graph before calling Varkeys.init_keys')

        for x_class, y_class in zip(x, y):

            _, embeddings, labels =  self._input_fn(x_class, y_class)   
            data.append((embeddings, labels))

            kmeans = KMeans(n_clusters=num_clusters, random_state=0, max_iter=self.kmeans_max_iter).fit(embeddings)
            centers = kmeans.cluster_centers_
            smart_keys.append(centers)

        smart_keys = np.concatenate(smart_keys, axis=0)
       
        self.sess.run(self.keys_init, feed_dict={self.keys_init_placeholder:smart_keys})
        del kmeans
        self.initialized = True
        return data


    def init_keys_random(self):
        # random_keys = tf.truncated_normal([self.keys_per_class*self.num_categories, self.keysize],mean=0, stddev=0.1)
        # random_keys = truncnorm.rvs(-0.1, 0.1, size=[self.keys_per_class*self.num_categories, self.keysize])
        # self.sess.run(self.keys_init, feed_dict={self.keys_init_placeholder:random_keys})
        self.keys = tf.Variable(tf.truncated_normal([n_keys, embedding_dim],mean=0, stddev=0.1), "keys")


    def init_keys_iterative_selection(self):
        # start with one key per class
        # after one epoch, batch_update, 
        # for each class
        #   take the instance with the lowest probability
        #   add this instance as new key
        self.max_keys_per_class = self.keys_per_class 
        self.keys_per_class = 1
        pass
    


    def keys_heatmap(self):
        keys = self.sess.run(self.keys)
        # normalize keys
        # normalized_keys = keys / np.linalg.norm(keys, axis=1).reshape(-1, 1)
        distances = dist(keys, keys)
        #assert (distances > 0).all()
        heatmap(distances)
        plt.show()
        #distances_ = self.kernel(keys, keys)
        #distances  = self.sess.run(distances_)
        # normalize
        #distances = distances / distances.max(axis=0)
        
        # plt.imshow(distances, cmap='hot', interpolation='nearest')
        # plt.imshow(distances, cmap='hot_r', interpolation=None)
        # plt.colorbar()
        # plt.show()
        # plt.clf()
        return

    def plot_keys(self):
        keys = self.sess.run(self.keys)
        keys_embedded = TSNE(n_components=2).fit_transform(keys)
        for i in range(self.num_categories):          
            start_index = i * self.keys_per_class
            end_index = (i+1) * self.keys_per_class
            keys_class_i = keys[start_index : end_index]
            plt.plot(keys_class_i[:, 0], keys_class_i[:, 1], 'o')
        plt.show()
        plt.clf()
        return

    def regularizer(self):
        # k_ij = Kernel(c_i, c_j) = 1 / d (inverse distance kernel)
        # for c_i = c_j, kernel(c_i, c_i) = some large number
        # we want the keys to be far apart
        # reg = sum( Kernel(c_i, c_j))
        return tf.reduce_sum(self.kernel(self.keys, self.keys))


    def update_keys(self, x, y, lr=0.01):
        
        keys = self.sess.run(self.keys)
        values = self.sess.run(self.values)
        h = self.sess.run(self.encoder, feed_dict={self.encoder_placeholder : x}) # shape: n x keysize

        def euclidean_distance(vector1, vector2):
            return np.sqrt(np.sum(np.power(vector1-vector2, 2)))

        def get_closest_keys(h, label):
            # h (=query) shape is (keysize, )

            keys_same = np.where((values==label).all(axis=1))[0] # indices of keys of same class than query h
            keys_other = np.where((values!=label).any(axis=1))[0]
            
            distances_same = []
            distances_other = []
            neighbors = []
            for i in range(0, keys.shape[0]):
                dist = euclidean_distance(h, keys[i, :])
                if i in keys_same:
                    distances_same.append((i, dist))
                else:
                    distances_other.append((i, dist))
            distances_same.sort(key=lambda x: x[1])
            distances_other.sort(key=lambda x: x[1])
            indices_same = [i for i, d in distances_same]
            indices_other = [i for i, d in distances_other]
            return indices_same[0], indices_other[0]
        
        for query, label in zip(h, y):
            # query shape is (keysize, )

            # find closest key to query for both groups:
            i, j = get_closest_keys(query, label)
            
            pos_key = keys[i]
            neg_key = keys[j]

            # update these two keys according to
            # loss = - || h - key_pos ||**2 + || h - key_neg ||**2
            # this memory loss is (mostly) concave, since on average the second distances should be larger
            # --> gradient ascent
            # k_pos = key_pos + lr * 2(h-key_pos) 
            # k_neg = key_neg - lr * 2(h-key_neg)
            pos_key = pos_key + lr * 2*(query-pos_key) /  np.sum( (pos_key - query)**2)
            neg_key = neg_key - lr * 2*(query-neg_key) /  np.sum( (neg_key - query)**2)


            # assign new value to keys tensor
            self.keys = tf.scatter_update(self.keys, [i], updates=pos_key.reshape(1, -1)) # pos_key
            self.keys = tf.scatter_update(self.keys, [j], updates=neg_key.reshape(1, -1)) # neg_key





if __name__ == '__main__':
    import matplotlib.pyplot as plt 
    num_classes = 10
    keys_per_class = 2
    dataset = 'cifar10'
    split_ratio = 0.2
    embedding_size = 2
    kmeans_max_iter = 100
    batch_size = 32
    sess = tf.Session()
    x_placeholder = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')
    y_placeholder = tf.placeholder(tf.float32, shape=(None, num_classes), name='output_y')
    encoder = conv_netV2(x_placeholder, embedding_size)
    m = Varkeys(method='knn', sess=sess, encoder=encoder, x_placeholder=x_placeholder, keysize=embedding_size, keys_per_class=keys_per_class, num_categories=num_classes, bandwidth=1, kmeans_max_iter=kmeans_max_iter)
    probas, v, hm = m(encoder)
    sess.run(tf.global_variables_initializer())
    x_train, x_val, x_test, y_train, y_val, y_test = get_dataset(dataset, ratio=split_ratio, normalize=True)
    data = m.init_keys_random()
    np.set_printoptions(threshold=sys.maxsize)
    minibatches = random_mini_batches(x_train, y_train, batch_size, 1)
    X, Y = minibatches[0]
    print(X.shape, Y.shape)
    print('encoder: ', encoder.shape)
    # print('probas: ', probas.shape)
    # print('values: ', v.shape)
    # print('hit_mask: ', hm.shape)
    #probas_, values_, hit_mask_, indices_ = sess.run([probas, values, hit_mask, indices], feed_dict={x_placeholder : X, y_placeholder : Y, m.get_bs_placeholder() : X.shape[0]})
    a = sess.run(encoder, feed_dict={x_placeholder : X, y_placeholder : Y, m.get_bs_placeholder() : X.shape[0]})
    print(a)
    # sys.stdout.flush()
    # b = sess.run(idx, feed_dict={x_placeholder : X, y_placeholder : Y, m.get_bs_placeholder() : X.shape[0]})
    # print(b)
    c = sess.run(hm, feed_dict={x_placeholder : X, y_placeholder : Y, m.get_bs_placeholder() : X.shape[0]})
    print(c)
    d = sess.run(v, feed_dict={x_placeholder : X, y_placeholder : Y, m.get_bs_placeholder() : X.shape[0]})
    print(d)
    
    sys.exit(0)
    
    print(m.keys)
    print(m.values)
    keys = sess.run(m.keys)
    V = sess.run(m.values)
    print(keys)
    print(V)
    colors = ['orange', 'yellow', 'green', 'blue', 'red', 'black', 'violet', 'brown', 'gray', 'lightblue'] 
    # for i in range(num_classes):
    #     x_class, y_class = data[i]
    #     start_index = i * keys_per_class
    #     end_index = (i+1) * keys_per_class
    #     keys_class_i = keys[start_index : end_index]
    #     plt.plot(keys_class_i[:, 0], keys_class_i[:, 1], 'ro', marker='x', markersize =10, color=colors[i])
        
    #     #xes
    #     print(y_class)
    #     print(y_class[0][i])
    #     assert y_class[0][i] == 1
    #     plt.plot(x_class[:, 0], x_class[:, 1], 'ro', marker='o', markersize =5, color=colors[i])
    #     plt.show()
    #     plt.clf()
    
    for i in range(num_classes):
        x_class, y_class = data[i]
        start_index = i * keys_per_class
        end_index = (i+1) * keys_per_class
        keys_class_i = keys[start_index : end_index]
        plt.plot(keys_class_i[:, 0], keys_class_i[:, 1], 'ro', marker='x', markersize =10, color=colors[i])
    plt.show()
    plt.clf()   
    sess.run(m.regularizer())
    m.keys_heatmap() 
    m.plot_keys() 







# bs = 10
# nkeys = 8
# classes = 4
# keys_per_class = 2
# k = 5

# import tensorflow as tf 
# import numpy as np 
# sess = tf.Session()
# hit_mask = tf.zeros([10, 8], dtype=tf.float32)
# idx = np.array([[ 0, 5, 7, 6, 7],
#                 [ 0, 5, 4, 3, 2],
#                 [ 0, 7, 6, 7, 4],
#                 [ 0, 3, 4, 1, 2],
#                 [ 0, 1, 2, 3, 4],
#                 [ 0, 5, 5, 3, 3],
#                 [ 7, 0, 6, 5, 4],
#                 [ 0, 1, 0, 0, 2],
#                 [ 3, 0, 2, 2, 5],
#                 [ 0, 1, 3, 4, 6]])
# indices = tf.constant(idx)
# indices2 = indices[:,:,None]
# a = tf.range(10)
# a = tf.reshape(a, [10, 1, 1])
# a = tf.tile(a, [1, 5, 1])
# indices3 = tf.concat([a, indices2], axis=2)
# updates = tf.ones([10,5])
# hit_mask_ = tf.tensor_scatter_nd_update(hit_mask, indices3, updates=updates)
# sess.run(hit_mask_)

# # shape = shape of output tensor (in this case same shape as input tensor)
# # indices.shape[-1] = shape.rank --> indices.shape[-1] = 2


# # min working example
# idx = np.array(   [[0, 1],
#                 [0, 4],
#                 [4, 3],
#                 [2, 1],
#                 [1, 0],
#                 [0, 4],
#                 [3, 2],
#                 [1, 3]])
# indices = tf.constant(idx)
# indices2 = indices[:,:,None]
# a = tf.range(8)
# a = tf.reshape(a, [8, 1, 1])
# a = tf.tile(a, [1, 2, 1])
# indices3 = tf.concat([a, indices2], axis=2)
# updates = tf.ones([8, 2])
# tensor = tf.zeros([8, 5])
# updated = tf.tensor_scatter_update(tensor, indices3, updates)

