import tensorflow as tf 
from model import conv_netV2, secondStage, SecondStage
import numpy as np 

import objgraph
import gc 


class Memory():

    def __init__(self,  encoder, encoder_input_placeholder, session, embedding_size=100, size=10000, target_size=10, K=50):
        if size < 0:
            raise('Size must be > 0!')
        if K > size:
            raise('Nearest neighbor parameter (K) must be smaller than memory size')
        self.capacity = size
        self.embedding_size = embedding_size
        self.target_size = target_size
        self.Keys   = tf.Variable(tf.zeros([self.capacity, self.embedding_size], dtype=tf.float32), dtype=tf.float32,  name='KEYS', trainable=False)
        self.Values = tf.Variable(tf.zeros([self.capacity, self.target_size], dtype=tf.float32), dtype=tf.float32, name='VALUES', trainable=False)
        self.K = K
        self.pointer = 0
        self.train_mode = True
        self.start_placeholder = tf.placeholder(tf.int32, shape=[])
        self.end_placeholder = tf.placeholder(tf.int32, shape=[])
        self.h_placeholder = tf.placeholder(tf.float32, shape=(None, self.embedding_size), name='h_placeholder')
        self.values_placeholder = tf.placeholder(tf.float32, shape=(None, self.target_size), name='values_placeholder')
        self.kernel_weights_placeholder = tf.placeholder(tf.float32, shape=(None,), name='kernel_weights_placeholder')
        self.encoder_input_placeholder = encoder_input_placeholder
        self.model = SecondStage(encoder=encoder, embedding_size=embedding_size, target_size=target_size)
            
        self.sess = session

        
    
    def initialize(self):
        self.model.set_session(self.sess)
        # build writer nodes and read nodes:
        self.updated_keys, self.updated_values = self.build_write()
        self.hit_keys, self.hit_values, self.kernel_weights = self.build_read()
        self.yhat, self.adapt_op, self.reset_ops, self.original_weights = self.build_adapt_predict()


    def training(self):
        self.train_mode = True 

    def eval(self):
        self.train_mode = False

    def free_space(self):
        return self.capacity - self.pointer

    def fit_into_memory(self, h):
        return h.shape[0] <= self.free_space()


    def build_write(self):
        with tf.name_scope('WRITE'):

            indices = tf.range(start=self.start_placeholder, limit=self.end_placeholder)
            self.updated_keys = tf.scatter_update(self.Keys, indices, updates=self.h_placeholder)
            self.updated_values = tf.scatter_update(self.Values, indices, updates=self.values_placeholder)
            return self.updated_keys, self.updated_values


    def write(self, h, values):
        # h: shape = (batch_size, embedding_size) 
        # values: shape = (batch_size, target_size)
        # we assume, capacity is a multiple of batch-size!!!!
        
        if self.pointer >= self.capacity:
            self.pointer = 0
            #print('reset pointer')
        # 1) if batch_size < free_space:
                # - end recursion
                # - fill free_space
        # 2) if batch_size > free_space:
                # - fill free_space
                # - recursive call: write(rest of h, rest of values)

        if self.fit_into_memory(h):
            self.sess.run([self.updated_keys, self.updated_values], feed_dict={self.start_placeholder : self.pointer,
                                                                               self.end_placeholder : self.pointer+h.shape[0],
                                                                               self.h_placeholder : h,
                                                                               self.values_placeholder : values})

            self.pointer += h.shape[0]

        else: # does not fit into memory
            free_space = self.free_space()
            offset = h.shape[0] - free_space
            h_tmp = h[:free_space]
            values_tmp = values[:free_space]
            self.sess.run([self.updated_keys, self.updated_values], feed_dict={self.start_placeholder : self.pointer,
                                                                               self.end_placeholder : self.capacity,
                                                                               self.h_placeholder : h_tmp,
                                                                               self.values_placeholder : values_tmp})
            self.pointer = 0
            # recursive call
            self.write(h[free_space:], values[free_space:])

    def build_read(self):
        with tf.name_scope('READ'):
            # keys: [capacity x embedding_size] -> [1 x capacity x embedding_size]
            expanded_keys = tf.expand_dims(self.Keys, axis=0) 

            # h: [batchsize x embedding_size] -> [batchsize x 1 x embedding_size]
            expanded_h = tf.expand_dims(self.h_placeholder, axis=1)

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

            weights = self.kernel(hit_keys, self.h_placeholder)

            return hit_keys, hit_values, weights


    def read(self, h):

        hit_keys, hit_values, weights = self.sess.run([self.hit_keys, self.hit_values, self.kernel_weights], feed_dict={self.h_placeholder : h})
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
        AB = AB[:,:,0] # last dim is just duplacates
        result = row_norms_A - 2 * AB + row_norms_B
        return result

    def kernel (self, A,B):
        #   1/(e + tf.square(hit_keys - h))
        distances = self.sq_distance(A,B)
        weights = tf.reciprocal(distances+tf.constant(1e-4))
        return weights # weight matrix: [batchsize x K]



    def build_adapt_predict(self, lr=0.01):
        with tf.variable_scope('ADAPT', reuse=tf.AUTO_REUSE):
            collections = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SECOND_STAGE')
            #original_weights = [tf.assign(tf.Variable(tf.zeros(layer.shape)), layer, validate_shape=False) for layer in collections]
            # weights_to_adapt = [tf.assign(tf.Variable(tf.zeros(layer.shape)), layer, validate_shape=False) for layer in collections]
            original_weights = [tf.placeholder(tf.float32, shape=layer.shape) for layer in collections]

            print(len(collections))
            print(len(original_weights))
           
            print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES ))

            # keys, values, weights = self.hit_keys, self.hit_values, self.kernel_weights
            print(self.hit_values.shape)
            print(self.kernel_weights.shape)

            logits_from_keys = self.model() # must get a placeholder containing the keys: model.get_h : keys
            cost = tf.reduce_sum(tf.losses.softmax_cross_entropy(logits=logits_from_keys, onehot_labels=self.values_placeholder, weights=self.kernel_weights_placeholder))
            reg = tf.reduce_sum([tf.reduce_sum(tf.square(collections[i] - original_weights[i])) for i in range(len(collections))])
            objective = cost + reg
            
            # adapter
            adapter = tf.train.AdamOptimizer(learning_rate=lr) # adapter and training optimizer both need to use weights of second stage network (--> reuse=True)
            adapt_op  = adapter.minimize(objective, var_list=collections)
            # gradients = adapter.compute_gradients(objective, var_list=weights_to_adapt)
            # adapt_op = adapter.apply_gradients(gradients)



            # reset adapted weights
            tmp =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SECOND_STAGE')
            print('variables to reset: ', tmp)
            reset_ops = []
            for i in range(len(tmp)):
                op = tf.assign(tmp[i], original_weights[i])
                #op = tmp[i].assign(original_weights[i])
                reset_ops.append(op)

        with tf.variable_scope('PREDICT'):
            # prediction:
            yhat = tf.nn.softmax(self.model())
        return yhat, adapt_op, reset_ops, original_weights


    def adapt_predict(self, h, keys, values, weights, niters=5):
        # h is a numpy array of shape [1 x embedding_size]
        #before = self.sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SECOND_STAGE'))[0]
        

        original_weights = self.sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SECOND_STAGE'))
        for _ in range(niters):
            
            nothing = self.sess.run(self.adapt_op, feed_dict={self.h_placeholder : h, 
                                                    self.model.get_h() : keys, 
                                                    self.values_placeholder : values, 
                                                    self.kernel_weights_placeholder : weights,
                                                    self.encoder_input_placeholder : np.zeros([32, 32, 32, 3]),
                                                    self.model.cond : 0,
                                                    **{self.original_weights[i] : original_weights[i] for i in range(len(original_weights))}
                                                    })
        
        #after = self.sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SECOND_STAGE'))[0]
        
        yhat = self.sess.run(self.yhat, feed_dict={
                                                    self.model.get_h() : h, 
                                                    self.encoder_input_placeholder : np.zeros([32, 32, 32, 3]),
                                                    self.model.cond : 0})
        for op in self.reset_ops:
            self.sess.run(op, feed_dict={self.original_weights[i] : original_weights[i] for i in range(len(original_weights))})
        #reset = self.sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SECOND_STAGE'))[0]
        
        # print('##################################################')
        # print((before==after).all()) # expect false
        # print((after == reset).all()) # expect false
        # print((before == reset).all()) # expect true
        # print('##################################################')
        
        return yhat



    def predict(self, hs):
        # hs is a numpy array
        
        hit_keys, hit_values, weights = self.read(hs)
        unstacked_keys = np.split(hit_keys, hit_keys.shape[0])
        unstacked_values = np.split(hit_values, hit_values.shape[0])
        unstacked_weights = np.split(weights, weights.shape[0])
        unstacked_h = np.split(hs, hs.shape[0])
        yhats = []
        for h, keys, values, weights in zip(unstacked_h, unstacked_keys, unstacked_values, unstacked_weights):
            # print(keys.shape) # (1, 50, 100)
            # print(values.shape) # (1, 50, 10)
            # print(weights.shape) # (1, 50)
            prediction = self.adapt_predict(h, keys[0], values[0], weights[0])
            yhats.append(prediction)
        return np.concatenate(yhats, axis=0)


























































































































class Memory_with_memroy_leak():

    def __init__(self, session, embedding_size=100, size=10000, target_size=10, K=50):
        if size < 0:
            raise('Size must be > 0!')
        if K > size:
            raise('Nearest neighbor parameter (K) must be smaller than memory size')
        self.capacity = size
        self.embedding_size = embedding_size
        self.target_size = target_size
        self.Keys   = tf.Variable(tf.zeros([self.capacity, self.embedding_size], dtype=tf.float32), dtype=tf.float32,  name='KEYS', trainable=False)
        self.Values = tf.Variable(tf.zeros([self.capacity, self.target_size], dtype=tf.float32), dtype=tf.float32, name='VALUES', trainable=False)
        self.K = tf.constant(K)
        self.pointer = 0
        self.train_mode = True
        self.model = SecondStage(embedding_size=embedding_size, target_size=target_size)
    
        self.session = session
        
    
    def initialize(self):
        self.model.set_session(self.session)


    def training(self):
        self.train_mode = True 

    def eval(self):
        self.train_mode = False

    def free_space(self):
        return self.capacity - self.pointer

    def fit_into_memory(self, h):
        return h.shape[0] <= self.free_space()


    def write(self, h, values):
        # h: shape = (batch_size, embedding_size) 
        # values: shape = (batch_size, target_size)
        # we assume, capacity is a multiple of batch-size!!!!
        
        if self.pointer >= self.capacity:
            self.pointer = 0
            #print('reset pointer')
        # 1) if batch_size < free_space:
                # - end recursion
                # - fill free_space
        # 2) if batch_size > free_space:
                # - fill free_space
                # - recursive call: write(rest of h, rest of values)

        if self.fit_into_memory(h):
            with tf.variable_scope('WRITE', reuse=tf.AUTO_REUSE):
                # end recursion:
                # indices = tf.Variable(tf.range(start=self.pointer, limit=self.pointer+h.shape[0]))
                indices = tf.range(start=self.pointer, limit=self.pointer+h.shape[0])
                #self.session.run(tf.variables_initializer(var_list=[indices]))
                self.Keys = tf.scatter_update(self.Keys, indices, updates=h)
                self.Values = tf.scatter_update(self.Values, indices, updates=values)
                self.pointer += h.shape[0]

        else: # does not fit into memory
            with tf.variable_scope('WRITE', reuse=tf.AUTO_REUSE):
                # indices = tf.Variable(tf.range(start=self.pointer, limit=self.capacity)) # tf.range does not include limit
                indices = tf.range(start=self.pointer, limit=self.capacity)
                #self.session.run(tf.variables_initializer(var_list=[indices]))
                free_space = self.free_space()
                offset = h.shape[0] - free_space
                h_tmp = h[:free_space]
                values_tmp = values[:free_space]
                print('partial h: ', h_tmp.shape, 'left_over: ', offset)
                self.Keys = tf.scatter_update(self.Keys, indices, updates=h_tmp)
                self.Values = tf.scatter_update(self.Values, indices, updates=values_tmp)
                self.pointer = 0

            # recursive call
            self.write(h[free_space:], values[free_space:])
    

    def read(self, h):

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
        AB = AB[:,:,0] # last dim is just duplacates
        result = row_norms_A - 2 * AB + row_norms_B
        return result

    def kernel (self, A,B):
        #   1/(e + tf.square(hit_keys - h))
        distances = self.sq_distance(A,B)
        weights = tf.reciprocal(distances+tf.constant(1e-4))
        return weights # weight matrix: [batchsize x K]

    def adapt_predict(self, h, niters=3, lr=0.001):
        # h: [1 x embedding_size]
        # niters: [1]
        with tf.variable_scope('READ', reuse=tf.AUTO_REUSE):
            keys, values, weights = self.read(h)
        with tf.variable_scope('ORIG', reuse=tf.AUTO_REUSE):
            collections = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SECOND_STAGE')
            original_weights = [tf.assign(tf.Variable(tf.zeros(layer.shape)), layer, validate_shape=False) for layer in collections]
        with tf.variable_scope('TMP', reuse=tf.AUTO_REUSE):
            weights_to_adapt = [tf.Variable(layer)  for layer in collections]
        
        
        logits = self.model(keys)
        cost = tf.reduce_sum(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=values, weights=weights))
        reg = tf.reduce_sum([tf.reduce_sum(tf.square(weights_to_adapt[i] - original_weights[i])) for i in range(len(original_weights))])
        objective = cost + reg
        
        #with tf.variable_scope('TMP', reuse=tf.AUTO_REUSE):
        adapter = tf.train.AdamOptimizer(learning_rate=lr)  
        #self.session.run(tf.variables_initializer(var_list=[adapter]))
        for _ in range(niters):
            gradients = adapter.compute_gradients(objective, var_list=weights_to_adapt)
            # for i, (grad, var) in enumerate(gradients):
            #     if grad is not None:
            #         gradients[i] = (tf.clip_by_norm(grad, tf.constant(1.0)), var)
            adapt_expr = adapter.apply_gradients(gradients)
        # predict
        yhat = tf.nn.softmax(self.model(h))

        # reset adapted weights
        tmp = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SECOND_STAGE')
        for i in range(len(tmp)):
            tmp[i]= tmp[i].assign(original_weights[i])
        del original_weights
        del weights_to_adapt
        return yhat



    def predict(self, hs):
        objgraph.show_most_common_types(limit=50)
        yhats = []
        #print('predicting')
        for h in tf.unstack(hs, axis=0):
            h = tf.expand_dims(h, axis=0)
            prediction = self.adapt_predict(h)
            yhats.append(prediction)
        gc.collect()
        return yhats
def predict(x_, tfsession):
    # x_: [batchsize x 32 x 32 x 3]

    hs_ = tfsession.run(embeddings, feed_dict={x:x_})
    yhats = M.predict(hs_)
    r =  tfsession.run(yhats, feed_dict={x:x_})
    return np.squeeze(np.array(r), axis=1)