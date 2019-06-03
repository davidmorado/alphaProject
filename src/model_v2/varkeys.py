import keras
from keras import Layer
import tensorflow as tf

class Varkeys(Layer):

    def __init__(self, keysize, keys_per_class, values, categories, bandwith, **kwargs):
        self.bandwith = bandwith
        self.initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=3, seed=None)
        self.categories = categories
        self.keysize = keysize
        self.dict_size = categories*keys_per_class
        self.values = tf.constant(np.repeat(np.eye(categories, dtype=int), keys_per_class, axis = 0), 
            dtype=tf.float32, shape = (self.dict_size, categories))
        super(Varkeys, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.keys = self.add_weight(name='keys', 
                                      shape=(self.dict_size, self.keysize),
                                      initializer=self.initializer,
                                      trainable=True)
        
        super(Varkeys, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        # switch out with KNN if you want!
        KV =  tf.matmul(tf.transpose(self.kernel(self.keys, x)), self.values)
        KV_ = tf.diag(tf.reshape(tf.reciprocal(tf.matmul(KV, tf.ones((self.categories,1)))), [-1]))
        output = tf.matmul(KV_, KV)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.categories)

    def get_memory(self):
        return self.keys
    
    def sq_distance(self, A, B):
        # print('im in distance function')
        row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
        row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

        row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
        row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

        return row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B


    def kernel (self, A,B):
        # print('im in kernel function!!')
        d = self.sq_distance(A,B)/self.bandwith
        o = tf.reciprocal(d+1e-4)
        return o


