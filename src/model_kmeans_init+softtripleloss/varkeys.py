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



class Varkeys:
    def __init__(self, sess, encoder, x_placeholder, keysize, keys_per_class, num_categories, bandwidth, kmeans_max_iter=5):
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


    def init_keys(self, x, y, data_ratio=0.2):
        num_clusters = self.keys_per_class
        

        x, y = self.sample(x, y, data_ratio)
        data = [] # for class in classes: (embeddings, labels)
        smart_keys = []

        if tf.report_uninitialized_variables().shape[0] > 0:
            raise Exception('Initialize Graph before calling Varkeys.init_keys')

        for x_class, y_class in zip(x, y):

            _, embeddings, labels =  self._input_fn(x_class, y_class)   
            data.append((embeddings, labels))

            kmeans = KMeans(n_clusters=num_clusters, random_state=0, max_iter=self.kmeans_max_iter).fit(embeddings)
            centers = kmeans.cluster_centers_
            smart_keys.append(centers)

        smart_keys = np.concatenate(smart_keys, axis=0)
       
        self.sess.run(self.keys_init, feed_dict={self.keys_init_placeholder:smart_keys})

        self.initialized = True
        return data


    def keys_heatmap(self):
        keys = self.sess.run(self.keys)
        # normalize keys
        # normalized_keys = keys / np.linalg.norm(keys, axis=1).reshape(-1, 1)
        distances = self.kernel(keys, keys)
        distances = self.sess.run(distances)

        # normalize
        distances = distances / distances.max(axis=0)
        
        # plt.imshow(distances, cmap='hot', interpolation='nearest')
        plt.imshow(distances, cmap='hot_r', interpolation='nearest')
        plt.colorbar()
        plt.show()
        plt.clf()


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

    def regularizer(self):
        # k_ij = Kernel(c_i, c_j) = 1 / d (inverse distance kernel)
        # for c_i = c_j, kernel(c_i, c_i) = some large number
        # we want the keys to be far apart
        # reg = sum( Kernel(c_i, c_j))
        return tf.reduce_sum(self.kernel(self.keys, self.keys))


# ValueError: Colormap cold is not recognized. Possible values are: 
# Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, 
# Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, 
# Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, 
# Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, 
# Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, 
# YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, 
# bone, bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, 
# cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, 
# gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, 
# gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, inferno, inferno_r, jet, jet_r, magma, magma_r, 
# nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, 
# seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, 
# terrain, terrain_r, twilight, twilight_r, twilight_shifted, twilight_shifted_r, viridis, viridis_r, winter, winter_r
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt 
    num_classes = 10
    keys_per_class = 3
    dataset = 'cifar10'
    split_ratio = 0.2
    embedding_size = 2
    kmeans_max_iter = 100
    sess = tf.Session()
    x_placeholder = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')
    y_placeholder = tf.placeholder(tf.float32, shape=(None, num_classes), name='output_y')
    encoder = conv_netV2(x_placeholder, embedding_size)
    m = Varkeys(sess=sess, encoder=encoder, x_placeholder=x_placeholder, keysize=embedding_size, keys_per_class=keys_per_class, num_categories=num_classes, bandwidth=1, kmeans_max_iter=kmeans_max_iter)
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





