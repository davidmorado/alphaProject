import tensorflow as tf 






class Memory():

    def __init__(self, model, embeddingsize=50, batch_size=128, capacity_multiplier=10, target_size=10):
        self.batch_size = batch_size
        self.capacity = batch_size * capacity_multiplier
        self.embeddingsize = embeddingsize
        self.target_size = target_size
        self.Keys   = tf.Variable(tf.zeros(self.capacity, self.embeddingsize), name='KEYS')
        self.Values = tf.Variable(tf.zeros(self.capacity, self.target_size), name='VALUES')
        self.K = 50
        self.pointer = 0
        self.train_mode = True
        self.model = model



    def training(self):
        self.train_mode = True 

    def eval(self):
        self.train_mode = False


    def in_train_phase(x, alt, training=None):
        """Selects `x` in train phase, and `alt` otherwise.
        Note that `alt` should have the *same shape* as `x`.
        Arguments:
            x: What to return in train phase
                (tensor or callable that returns a tensor).
            alt: What to return otherwise
                (tensor or callable that returns a tensor).
            training: Optional scalar tensor
                (or Python boolean, or Python integer)
                specifying the learning phase.
        Returns:
            Either `x` or `alt` based on the `training` flag.
            the `training` flag defaults to `K.learning_phase()`.
        """
        #   if training is None:
        #     training = learning_phase()

        if training == 1 or training is True:
            if callable(x):
            return x()
            else:
            return x

        elif training == 0 or training is False:
            if callable(alt):
            return alt()
            else:
            return alt    


    def write(h, values):
        # h: shape = (batch_size, embeddingsize) 
        # values: shape = (batch_size, target_size)
        # we assume, capacity is a multiple of batch-size!!!!

        if self.pointer >= self.capacity:
            self.pointer = 0

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
        # hit_keys: [K x embeddingsize]
        # hit_values: [K x targetsize]
        hit_keys = tf.nn.embedding_lookup(self.Keys, indices)
        hit_values = tf.nn.embedding_lookup(self.Values, indices)

        weights = self.kernel(hit_keys, h)

        return hit_keys, hit_values, weights

    def sq_distance(self, A, B):
        # A = hit_keys: [batchsize x K x embeddingsize]
        # B = h: [batchsize x embeddingsize]
        # computes ||A||^2 - 2*||AB|| + ||B||^2 = A.TA - 2 A.T B + B.T B
        row_norms_A = tf.reduce_sum(tf.square(A), axis=2) 
        row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
        row_norms_B = tf.reshape(row_normsB, [-1, 1])
        # B: [batchsize x embeddingsize x 1]
        B = tf.expanddims(B, axis=2)
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
    	#	1/(e + tf.square(hit_keys - h))
        distances = self.sq_distance(A,B)
        weights = tf.reciprocal(distances+tf.constant(1e-4))
        return weights # weight matrix: [K x batchsize]

    def adaptation(self, h,  niters):

        keys, values, weights = self.read(h)

        cost = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=y)
    
        weights_to_adapt = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'SECOND_STAGE'
            )
        delta_total = [tf.Variable(tf.zeros(weight.shape), name='delta_total') for weight in weights_to_adapt]
        for step in range(niters):


        gradients = optimizer.compute_gradients(error, var_list=weights_to_adapt)


########################################################################################################
encode_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, '{}/encode'.format(scope)
            )


trained_vars = encode_vars
        for i in range(num_actions):
            trained_vars += tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, 'dnd{}/KEYS'.format(i))
        gradients = optimizer.compute_gradients(error, var_list=trained_vars)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, grad_clipping), var)
        optimize_expr = optimizer.apply_gradients(gradients)



capacity = 10
bs = 5


p = 0

0
1
2
3
4

p = 5 (p = p + bs)

5
6
7
8
9

p = 10
capacity = 10























