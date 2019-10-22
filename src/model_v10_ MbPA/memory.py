import tensorflow as tf 
from models import Stage1, Stage2



class memory():

    def __init__(self, embedding_size=100, capacity=100, target_size=10, K=50):
        if capacity <= 0:
            raise('Capacity must be > 0!')
        if K >= capacity:
            raise('Nearest neighbor parameter (K) must be smaller than capacity')
        self.capacity = capacity
        self.embedding_size = embedding_size
        self.target_size = target_size
        self.Keys   = tf.Variable(tf.zeros([self.capacity, self.embedding_size], dtype=tf.float32), dtype=tf.float32,  name='memory/KEYS', trainable=False)
        self.Values = tf.Variable(tf.zeros([self.capacity, self.target_size], dtype=tf.float32), dtype=tf.float32, name='memory/VALUES', trainable=False)
        self.K = tf.constant(K)
        self.pointer = 0
        self.train_mode = True
        # self.model = Stage2(embedding_size=embedding_size, target_size=target_size)
        # self.session = session
        
    # def initialize(self):
    #     self.model.set_session(self.session)


    # def training(self):
    #     self.train_mode = True 

    # def eval(self):
    #     self.train_mode = False

    def free_space(self):
        return self.capacity - self.pointer

    def fit_into_memory(self, h):
        return h.shape[0] <= self.free_space()


    def write(self, h, values):
        # h: shape = (batch_size, embedding_size) 
        # values: shape = (batch_size, target_size)

        bs = tf.shape(h)[0]

        with tf.variable_scope('memory/WRITE', reuse=tf.AUTO_REUSE):

            if self.fit_into_memory(h):
                indices = tf.range(start=self.pointer, limit=self.pointer+bs)
                self.Keys = tf.scatter_update(self.Keys, indices, updates=h)
                self.Values = tf.scatter_update(self.Values, indices, updates=values)
                self.pointer += bs

            else: 
                # does not fit into memory
                indices = tf.range(start=self.pointer, limit=self.capacity)
                free_space = self.free_space()
                offset = bs - free_space
                self.Keys = tf.scatter_update(self.Keys, indices, updates=h[:free_space])
                self.Values = tf.scatter_update(self.Values, indices, updates=values[:free_space]) 
                self.pointer = 0
                # write remaining
                indices = tf.range(start=self.pointer, limit=offset)
                self.Keys = tf.scatter_update(self.Keys, indices, updates=h[free_space:])
                self.Values = tf.scatter_update(self.Values, indices, updates=values[free_space:])
                self.pointer += offset

    def read(self, h):

        with tf.variable_scope('memory/READ', reuse=tf.AUTO_REUSE):

            # keys: [capacity x embedding_size] -> [1 x capacity x embedding_size]
            expanded_keys = tf.expand_dims(self.Keys, axis=0) 

            # h: [batchsize x embedding_size] -> [batchsize x 1 x embedding_size]
            expanded_h = tf.expand_dims(h, axis=1)

            # h: [batchsize x 1 x embedding_size] -> [batchsize x capacity x embedding_size]
            tiled_eh = tf.tile(expanded_h, [1, self.capacity, 1])

            # keys - h: [batchsize x capacity x embedding_size]
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
            # hit_keys: [K x embedding_size]
            # hit_values: [K x targetsize]
            hit_keys = tf.nn.embedding_lookup(self.Keys, indices)
            hit_values = tf.nn.embedding_lookup(self.Values, indices)

            weights = self.kernel(hit_keys, h)

        return hit_keys, hit_values, weights

    def sq_distance(self, A, B):
        # A = hit_keys: [batchsize x K x embedding_size]
        # B = h: [batchsize x embedding_size]
        # computes ||A||^2 - 2*||AB|| + ||B||^2 = A.TA - 2 A.T B + B.T B
        row_norms_A = tf.reduce_sum(tf.square(A), axis=2)
        row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
        row_norms_B = tf.reshape(row_norms_B, [-1, 1])
        # B: [batchsize x embedding_size x 1]
        B = tf.expand_dims(B, axis=2)
        # B: [batchsize x embedding_size x K]
        B = tf.tile(B, [1, 1, self.K])
        # https://stackoverflow.com/questions/38235555/tensorflow-matmul-of-input-matrix-with-batch-data
        # AB = [batchsize x K x embedding_size] @ [batchsize x embedding_size x K] 
        # -> [batchsize x K x K] (duplicated on axis 2)
        AB = tf.matmul(A, B) 
        # AB -> [batchsize x K]
        AB = AB[:,:,0] # last dim is just duplicates
        result = row_norms_A - 2 * AB + row_norms_B
        return result

    def kernel (self, A,B):
        #   1/(e + tf.square(hit_keys - h))
        distances = self.sq_distance(A,B)
        weights = tf.reciprocal(distances+tf.constant(1e-4))
        return weights # weight matrix: [batchsize x K]

    # def adapt_predict(self, h, niters=3, lr=0.001):
    #     # h: [1 x embedding_size]
    #     # niters: [1]
    #     with tf.variable_scope('READ', reuse=tf.AUTO_REUSE):
    #         keys, values, weights = self.read(h)
    #     with tf.variable_scope('ORIG', reuse=tf.AUTO_REUSE):
    #         collections = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SECOND_STAGE')
    #         original_weights = [tf.assign(tf.Variable(tf.zeros(layer.shape)), layer, validate_shape=False) for layer in collections]
    #     with tf.variable_scope('TMP', reuse=tf.AUTO_REUSE):
    #         weights_to_adapt = [tf.Variable(layer)  for layer in collections]
        
        
    #     logits = self.model(keys)
    #     cost = tf.reduce_sum(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=values, weights=weights))
    #     reg = tf.reduce_sum([tf.reduce_sum(tf.square(weights_to_adapt[i] - original_weights[i])) for i in range(len(original_weights))])
    #     objective = cost + reg
        
    #     #with tf.variable_scope('TMP', reuse=tf.AUTO_REUSE):
    #     adapter = tf.train.AdamOptimizer(learning_rate=lr)  
    #     #self.session.run(tf.variables_initializer(var_list=[adapter]))
    #     for _ in range(niters):
    #         gradients = adapter.compute_gradients(objective, var_list=weights_to_adapt)
    #         # for i, (grad, var) in enumerate(gradients):
    #         #     if grad is not None:
    #         #         gradients[i] = (tf.clip_by_norm(grad, tf.constant(1.0)), var)
    #         adapt_expr = adapter.apply_gradients(gradients)
    #     # predict
    #     yhat = tf.nn.softmax(self.model(h))

    #     # reset adapted weights
    #     tmp = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SECOND_STAGE')
    #     for i in range(len(tmp)):
    #         tmp[i]= tmp[i].assign(original_weights[i])
    #     del original_weights
    #     del weights_to_adapt
    #     return yhat



    # def predict(self, hs):
    #     objgraph.show_most_common_types(limit=50)
    #     yhats = []
    #     #print('predicting')
    #     for h in tf.unstack(hs, axis=0):
    #         h = tf.expand_dims(h, axis=0)
    #         prediction = self.adapt_predict(h)
    #         yhats.append(prediction)
    #     gc.collect()
    #     return yhats




















