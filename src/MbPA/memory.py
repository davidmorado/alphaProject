import tensorflow as tf 






class Memory():

    def __init__(self, embeddingsize=50, batch_size=128, capacity_multiplier=10, target_size=10):
        self.batch_size = batch_size
        self.capacity = batch_size * capacity_multiplier
        self.embeddingsize = embeddingsize
        self.target_size = target_size
        self.Keys   = tf.Variable(tf.zeros(self.capacity, self.embeddingsize), name='KEYS')
        self.Values = tf.Variable(tf.zeros(self.capacity, self.target_size), name='VALUES')
        self.K = 50
        self.pointer = 0


    def write(hs, values):
        # hs: shape = (batch_size, embeddingsize) 
        # values: shape = (batch_size, target_size)
        # we assume, capacity is a multiple of batch-size!!!!

        if self.pointer >= self.capacity:
            self.pointer = 0

        indices = tf.Variable(tf.range(start=self.pointer, limit=self.pointer+self.batch_size))
        tf.scatter_update(self.Keys, indices, updates=hs)
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

    	# sum(diff^2): [batchsize x capacity]
    	distances = tf.reduce_sum(
                tf.square(diff),
                axis=2
            )

    	# negate distances to get the k closest keys
    	# indices: [K] 
        _, indices = tf.nn.top_k(-distances, k=self.K)

        # lookup
        # hit_keys: [K x embeddingsize]
        # hit_values: [K x targetsize]
        hit_keys = tf.nn.embedding_lookup(self.Keys, indices)
        hit_values = tf.nn.embedding_lookup(self.Values, indices)

        weights = self.kernel(hit_keys, h)
        return hit_keys, hit_values, weights

    
    def sq_distance(self, A, B):
    	# A = hit_keys: [K x embeddingsize]
    	# B = h: [batchsize x embeddingsize]
    	# computes ||A||^2 - 2*||AB|| + ||B||^2 = A.TA - 2 A.T B + B.T B
        row_norms_A = tf.reduce_sum(tf.square(A), axis=1) # shape (K)
        row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # shape (K x 1)

        row_norms_B = tf.reduce_sum(tf.square(B), axis=1) # shape (batchsize)
        row_norms_B = tf.reshape(row_norms_B, [1, -1])  # shape (batchsize x 1)

        AB = tf.matmul(A, tf.transpose(B)) # shape (K x batchsize)

        return row_norms_A - 2 * AB + row_norms_B


    def kernel (self, A,B):
    	#	1/(e + tf.square(hit_keys - h))
        distances = self.sq_distance(A,B)
        weights = tf.reciprocal(distances+1e-4)
        return weights # weight matrix: [K x batchsize]










def _build_reader(self, h, epsize):
        ''' a fucntion to build reader network. 
        this function will only be called once.
        ARGS:
            h: encoded states. its' shape is (batch_size, keysize)
        '''
        with tf.name_scope('lookup'):
            # only take into account the current epsize
            # both shapes are (epsize, keysize)
            keys = self.memory_keys[:epsize]
            values = self.memory_values[:epsize]

            # set both shapes to (batch_size, epsize, keysize)
            # and compute distances
            # [keys].shape: (1, epsize, keysize)
            # tf.shape(h): batchsize
            # tiled_keys = tf.tile([keys], [tf.shape(h)[0], 1, 1])
            # expanded_keys: (1, epsize, keys)
            expanded_keys = tf.expand_dims(keys, axis=0)

            # h.shape: (batchsize, keysize)
            expanded_h = tf.expand_dims(h, axis=1)
            # expanded_h.shape: (batch_size, 1, keysize)
            # tiled_eh.shape: (batch_size, epsize, keysize)
            tiled_eh = tf.tile(expanded_h, [1, epsize, 1])

            # compute distances
            distances = tf.reduce_sum(
                tf.square(expanded_keys - tiled_eh),
                axis=2
            )

            # negate distances to get the k closest keys
            _, indices = tf.nn.top_k(-distances, k=self.p)  # indecies (?, 10)

            # distances (?, ?) batchsize, memsize
            # get p distances
            hit_keys = tf.nn.embedding_lookup(keys, indices)
            hit_values = tf.nn.embedding_lookup(values, indices)
            flatten_indices = tf.reshape(indices, [-1])  # batch * self.p
            unique_indicies, _ = tf.unique(flatten_indices)
            update_ages = tf.group(*[
                # increment ages
                tf.assign(self.memory_ages, self.memory_ages + 1),
                # reset hit ages
                tf.scatter_update(
                    self.memory_ages, unique_indicies,
                    tf.zeros([self.p], dtype=tf.float32)
                )
                # tf.assign(hit_ages, 0)
            ])
        return hit_keys, hit_values, update_ages










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























