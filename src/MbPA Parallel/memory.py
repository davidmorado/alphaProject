import tensorflow as tf 
from utils import in_train_phase
import sys




class Memory():

    def __init__(self, model, embeddingsize=50, batch_size=128, capacity_multiplier=10, target_size=10, K=20):
        self.batch_size = batch_size
        self.capacity = batch_size * capacity_multiplier
        self.embeddingsize = embeddingsize
        self.target_size = target_size
        self.Keys   = tf.Variable(tf.zeros([self.capacity, self.embeddingsize], dtype=tf.float32), dtype=tf.float32,  name='KEYS')
        self.Values = tf.Variable(tf.zeros([self.capacity, self.target_size], dtype=tf.float32), dtype=tf.float32, name='VALUES')
        self.K = tf.constant(K)
        self.pointer = 0
        self.train_mode = True
        self.model = model



    def training(self):
        self.train_mode = True 

    def eval(self):
        self.train_mode = False


    def write(self, h, values):
        # h: shape = (batch_size, embeddingsize) 
        # values: shape = (batch_size, target_size)
        # we assume, capacity is a multiple of batch-size!!!!

        if self.pointer >= self.capacity:
            self.pointer = 0
            print('reset pointer')

        indices = tf.Variable(tf.range(start=self.pointer, limit=self.pointer+self.batch_size))
        tf.scatter_update(self.Keys, indices, updates=h)
        tf.scatter_update(self.Values, indices, updates=values)

        self.pointer += self.batch_size

    def read(self, h):

        # keys: [capacity x embeddingsize] -> [1 x capacity x embeddingsize]
        expanded_keys = tf.expand_dims(self.Keys, axis=0) 

        # h: [batchsize x embeddingsize] -> [batchsize x 1 x embeddingsize]
        expanded_h = tf.expand_dims(h, axis=1)

        # h: [batchsize x 1 x embeddingsize] -> [batchsize x capacity x embeddingsize]
        tiled_eh = tf.tile(expanded_h, [1, self.capacity, 1])

        # keys - h: [batchsize x capacity x embeddingsize]
        diff = expanded_keys - tiled_eh

        # distances: [batchsize x capacity]
        distances = tf.reduce_sum(
                tf.square(diff),
                axis=2
            )

        # negate distances to get the k closest keys
        # indices: [querybatchsize x K] 
        _, indices = tf.nn.top_k(-distances, k=self.K)

        # lookup of 
        # hit_keys: [batchsize x K x embeddingsize]
        # hit_values: [batchsize x K x targetsize]
        hit_keys = tf.nn.embedding_lookup(self.Keys, indices)
        hit_values = tf.nn.embedding_lookup(self.Values, indices)

        # [batchsize x K]
        weights = self.kernel(hit_keys, h)

        return hit_keys, hit_values, weights

    def sq_distance(self, A, B):
        # A = hit_keys: [batchsize x K x embeddingsize]
        # B = h: [batchsize x embeddingsize]
        # computes ||A||^2 - 2*||AB|| + ||B||^2 = A.TA - 2 A.T B + B.T B
        row_norms_A = tf.reduce_sum(tf.square(A), axis=2)
        row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
        row_norms_B = tf.reshape(row_norms_B, [-1, 1])
        # B: [batchsize x embeddingsize x 1]
        B = tf.expand_dims(B, axis=2)
        # B: [batchsize x embeddingsize x K]
        B = tf.tile(B, [1, 1, self.K])
        # https://stackoverflow.com/questions/38235555/tensorflow-matmul-of-input-matrix-with-batch-data
        # AB = [batchsize x K x embeddingsize] @ [batchsize x embeddingsize x K] 
        # -> [batchsize x K x K] (duplicated on axis 2)
        AB = tf.matmul(A, B) 
        # AB -> [batchsize x K]
        AB = AB[:,:,0] # last dim is just duplacates
        result = row_norms_A - 2 * AB + row_norms_B
        return result

    def kernel (self, A,B):
        #   1/(e + tf.square(hit_keys - h))
        distances = self.sq_distance(A,B)
        weights = tf.reciprocal(distances+tf.constant(1e-4))
        return weights # weight matrix: [batchsize x K]

    def adapt_predict(self, h, niters=10, lr=0.001):
        # h: [N x embeddingsize]
        # niters: [1]

        keys, values, weights = self.read(h)
    
        collections = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SECOND_STAGE')
        original_weights = [tf.assign(tf.Variable(tf.zeros(layer.shape), dtype=tf.float32), layer, validate_shape=False) for layer in collections]
        tiled_orig_weights = [tf.tile(tf.expand_dims(layer, axis=0), tf.scatter_update(tf.Variable(tf.ones(tf.rank(layer)+1, dtype=tf.int32)), tf.constant([0]), h.shape[0])) for layer in original_weights]      
        weights_to_adapt = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SECOND_STAGE')
        tiled_weights_to_adapt = [tf.tile(tf.expand_dims(layer, axis=0), tf.scatter_update(tf.Variable(tf.ones(tf.rank(layer)+1, dtype=tf.int32)), tf.constant([0]), h.shape[0])) for layer in weights_to_adapt]

        # tiled_orig_weights = [tf.cond(tf.rank(layer) > 2, tf.tile(tf.expand_dims(layer, axis=0), [h.shape[0], 1, 1]), tf.tile(tf.expand_dims(layer, axis=0), [h.shape[0], 1])) for layer in original_weights]      
        # weights_to_adapt = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SECOND_STAGE')
        # tiled_weights_to_adapt = [tf.tile(tf.expand_dims(layer, axis=0), [h.shape[0], 1, 1]) for layer in weights_to_adapt]

        # creates list of len batchsize with elements with [K x embeddingsize]
        keys = tf.unstack(keys)
        # values:[K x embeddingsize] -> logits:[K x targetsize]
        logits = [self.model(key) for key in keys]
        logits = tf.stack(logits) # [batchsize x K x targetsize]
        cost = tf.reduce_sum(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=values, weights=weights, reduction=tf.losses.Reduction.NONE), axis=1)
        print(tiled_weights_to_adapt[0])
        print(tiled_orig_weights[0])
        reg = tf.reduce_sum([tf.reduce_sum(tf.square(tiled_weights_to_adapt[i] - tiled_orig_weights[i]), axis=tf.range(1, tf.rank(tiled_weights_to_adapt[i]))) for i in range(len(original_weights))], axis=0)
        print(cost)
        print(reg)
        objective = cost + reg
         
        with tf.variable_scope('TMP', reuse=tf.AUTO_REUSE):
            adapter = tf.train.AdamOptimizer(learning_rate=lr)

            with tf.Session() as sess:
                # adapt weights
                for step in range(niters):
                    gradients = adapter.compute_gradients(objective, var_list=tiled_weights_to_adapt)
                    # for i, (grad, var) in enumerate(gradients):
                        # if grad is not None:
                        #     gradients[i] = (tf.clip_by_norm(grad, tf.constant(1.0)), var)
                    print(gradients)
                    adapt_expr = adapter.apply_gradients(gradients)
                # predict
                yhat = tf.nn.softmax(self.model(h))

            # reset adapted weights
            tmp = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SECOND_STAGE')
            for i in range(len(tmp)):
                tmp[i]= tmp[i].assign(original_weights[i])

        return yhat



    def predict(self, hs):

        yhats = []
        for h in tf.unstack(hs, axis=0):
            h = tf.expand_dims(h, axis=0)
            yhats.append(self.adapt_predict(h))

        return yhats



















