
            #return hit_keys, hit_values
            # mask of zeros and ones of hit keys and hit values
            
            #hit_mask = tf.Variable(tf.zeros([bs, dict_size], dtype=tf.float32), dtype=tf.float32,  name='hit_mask', trainable=False)
            # hit_mask = tf.Variable(tf.fill([self.bs, dict_size], value=0.0), dtype=tf.float32,  name='hit_mask', trainable=False)
            # hit_mask = tf.scatter_update(hit_mask, indices, updates=1) # [batchsize x n_keys]
            hit_mask = tf.zeros([self.bs, dict_size], dtype=tf.float32)         # [batchsize x n_keys]

            
            # tf.gather_nd(hit_mask, indices)
            # with tf.Session() as sess:
            #     print(sess.run([tf.rank(hit_mask), tf.rank(indices), tf.rank(tf.ones([self.bs, self.k]))]))
            hit_mask_ = tf.tensor_scatter_nd_update(hit_mask, [indices], updates=tf.ones([self.bs, self.k])) # [batchsize x n_keys]
            print(hit_mask_)
            
            values_expanded = tf.multiply(tf.tile(tf.expand_dims(hit_mask_, axis=2), [1, 1, num_classes]) , values_expanded) # elementwise multiplication
            print('values: ', values_expanded)
            return keys_expanded, values_expanded, hit_mask_, indices