
import tensorflow as tf
from model import secondStage

class Memory():

    def __init__(self, model, embeddingsize=50, batch_size=128, capacity_multiplier=10, target_size=10, K=20):
        self.batch_size = batch_size
        self.capacity = batch_size * capacity_multiplier
        self.embeddingsize = embeddingsize
        print(self.embeddingsize)
        print(self.capacity)
        self.target_size = target_size
        self.Keys   = tf.Variable(tf.zeros([self.capacity, self.embeddingsize]),dtype=tf.float32 , name='KEYS')
        self.Values = tf.Variable(tf.zeros([self.capacity, self.target_size]), dtype=tf.float32, name='VALUES')
        self.K = tf.constant(K)
        self.pointer = tf.Variable(0)
        self.train_mode = True
        self.model = model

M = Memory(secondStage)
































# def sq_distance(A, B):
#     print(A, 'batchsize x K x embeddingsize')
#     print(B, 'batchsize x embeddingsize x K')
#     row_norms_A = tf.reduce_sum(tf.square(A), axis=2) 
#     #row_norms_A = tf.reshape(row_norms_A, [-1, 1])
#     row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
#     row_norms_B = tf.reshape(row_normsB, [-1, 1])
#     B = tf.expanddims(B, axis=2)
#     B = tf.tile(B, [1, 1, k])
#     print('B ', B)
#     AB = tf.matmul(A, B) # batchsize x K x K,  # https://stackoverflow.com/questions/38235555/tensorflow-matmul-of-input-matrix-with-batch-data
#     AB = AB[:,:,0] # last dim is just duplacates
#     print('row_norms_A ', row_norms_A)
#     print('row_norms_B ', row_norms_B)
#     print('AB', AB)
#     result = row_norms_A - 2 * AB + row_norms_B
#     print('result', result)
#     return result
# Stack Overflow
# Tensorflow - matmul of input matrix with batch data
# I have some data represented by input_x. It is a tensor of unknown size (should be inputted by batch) and each item there is of size n. input_x undergoes tf.nn.embedding_lookup, so that embed now has

# def sq_distance(A, B):
#     print(A, 'batchsize x K x embeddingsize')
#     print(B, 'batchsize x embeddingsize x K')
#     row_norms_A = tf.reduce_sum(tf.square(A), axis=2) 
#     #row_norms_A = tf.reshape(row_norms_A, [-1, 1])
#     row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
#     row_norms_B = tf.reshape(row_normsB, [-1, 1])
#     B = tf.expanddims(B, axis=2)
#     B = tf.tile(B, [1, 1, k])
#     print('B ', B)
#     AB = tf.matmul(A, B) # batchsize x K x K,  # https://stackoverflow.com/questions/38235555/tensorflow-matmul-of-input-matrix-with-batch-data
#     AB = AB[:,:,0] # last dim is just duplacates
#     print('row_norms_A ', row_norms_A)
#     print('row_norms_B ', row_norms_B)
#     print('AB', AB)
#     result = row_norms_A - 2 * AB + row_norms_B
#     print('result', result)
#     return result



# def kernel (A,B):
#     distances = sq_distance(A,B)
#     weights = tf.reciprocal(distances+tf.constant(1e-4))
#     return weights



# k = 3
# bs = 2
# embsize = 3
# capacity = 5

# import tensorflow as tf
# h = tf.constant([[0, 0.0, 0], [4, 4, 4]])                             # batchsize x embeddingsize (2 x 3
# keys = tf.constant([[1, 5, 1],[2,5,3], [3,6.0,3], [4,4,1], [5, 5., 1]]) # capacity x embeddingsize (5 x 3)
# sess = tf.Session()
# expanded_keys = tf.expand_dims(keys, axis=0) 
# expanded_h = tf.expand_dims(h, axis=1)
# tiled_eh = tf.tile(expanded_h, [1, capacity, 1])
# diff = expanded_keys - tiled_eh
# distances = tf.reducesum(tf.square(diff), axis=2)
# , indices = tf.nn.top_k(-distances, k=k)
# hit_keys = tf.nn.embedding_lookup(keys, indices)
# hit_keys
# weights = kernel(hit_keys, h)

# sess.run(distances)
# sess.run(indices)
# sess.run(hit_keys)
# sess.run(weights)






# # import tensorflow as tf 
# # distances = tf.constant([[1, 2, 3, 7], [3, 4, 5, 6], [2, 4, 6, 4]]) # batchsize x capacity
# # keys = tf.constant([[1, 5, 1],[2,5,3], [3,6,3], [4,4,1]])           # memorysize x embeddingsize
# # values = tf.constant([[0, 1], [0, 1], [1, 0]])						# batchsize x targetsize
# # sess = tf.Session()
# # _, idx = tf.nn.top_k(-distances, 2)                                     # batchsize x K
# # a =  tf.nn.embedding_lookup(keys, idx)                                # batchsize x K x embeddingsize
# # b =  tf.nn.embedding_lookup(values, idx)                                # batchsize x K x embeddingsize

# # sess.run(distances)
# # sess.run(idx)
# # sess.run(keys)
# # sess.run(a)
# # sess.run(b)



# # import tensorflow as tf
# # import numpy as np


# # def sq_distance(A, B):
# # 	# A = hit_keys: [batchsize x K x embeddingsize]
# # 	# B = h: [batchsize x embeddingsize]
# # 	# computes ||A||^2 - 2*||AB|| + ||B||^2 = A.TA - 2 A.T B + B.T B
# #     row_norms_A = tf.reduce_sum(tf.square(A), axis=2) # shape (batchsize x K)
# #     row_norms_A = tf.expand_dims(row_norms_A, axis=2)  # shape (batchsize x K x 1)
# #     # row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # shape (batchsize x K x 1)

# #     row_norms_B = tf.reduce_sum(tf.square(B), axis=1) # shape (batchsize)
# #     row_norms_B = tf.reshape(row_norms_B, [1, -1])  # shape (batchsize x 1)

# #     # AB = tf.matmul(A, tf.expand_dims(tf.transpose(B), axis=1)) 
# #     print('A & B')
# #     print(A)
# #     print(B)
# #     eB = tf.expand_dims(B, axis=2)
# #     tiled_B = tf.tile(eB, [1, 1, 2])
# #     print(tiled_B)
# #     # AB = tf.matmul(A, tiled_B)
# #     AB = tf.tensordot(A, tiled_B, axes = [[1, 2], [1, 2]]) # shape (batchsize x K)
# #     print('AB')
# #     print(AB)
# #     #  [3,2,4] @ [3, 4] -> [3, 2]

# #     return row_norms_A - 2 * AB + row_norms_B


# # def kernel(A, B):
# # 	#	1/(e + tf.square(hit_keys - h))
# #     distances = sq_distance(A,B)
# #     print(distances)
# #     weights = tf.reciprocal(distances+tf.constant(1e-4))
# #     return weights # weight matrix: [K x batchsize]

# # capacity = 5

# # # memory [capacity x embeddingsize]
# # Keys = tf.constant([[1,7,5,4],
# # 					  [3,3,3,3],
# # 					  [1,2,3,4],
# # 					  [9,8,7,5],
# # 					  [4,2,6,4]], dtype=np.float32)

# # # h [qbatchsize x embeddingsize]
# # h = tf.constant([[1,7,5,4],
# # 				  [1,2,3,4],
# # 				  [4,2,6,4]], dtype=np.float32)

# # Values = tf.constant([[1,0,0,0,0], 
# # 					  [0,1,0,0,0], 
# # 					  [0,0,1,0,0],
# # 					  [0,0,0,1,0],
# # 					  [0,0,0,0,1]], dtype=np.float32)

# # # keys: [capacity x embeddingsize] -> [1 x capacity x embeddingsize]
# # expanded_keys = tf.expand_dims(Keys, axis=0) 

# # # h: [batchsize x embeddingsize] -> [batchsize x 1 x embeddingsize]
# # expanded_h = tf.expand_dims(h, axis=1)

# # # h: [batchsize x 1 x embeddingsize] -> [batchsize x capacity x embeddingsize]
# # tiled_eh = tf.tile(expanded_h, [1, capacity, 1])

# # # keys - h: [batchsize x capacity x embeddingsize]
# # diff = expanded_keys - tiled_eh

# # # distances: [batchsize x capacity]
# # distances = tf.reduce_sum(
# #         tf.square(diff),
# #         axis=2
# #     )

# # # negate distances to get the k closest keys
# # # indices: [querybatchsize x K] 
# # _, indices = tf.nn.top_k(-distances, k=2)

# # # lookup of 
# # # hit_keys: [batchsize x K x embeddingsize]
# # # hit_values: [batchsize x K x targetsize]
# # # hit_keys   = tf.gather(Keys, indices)
# # # hit_values = tf.gather(Values, indices)
# # hit_keys   = tf.nn.embedding_lookup(Keys, indices)
# # hit_values = tf.nn.embedding_lookup(Values, indices)

# # weights = kernel(hit_keys, h) # [batchsize x K]

# # sess = tf.Session()
# # sess.run(hit_keys)
# # sess.run(hit_values)
# # sess.run(weights)










# # weights = self.kernel(hit_keys, h)





# # # y [qbatchsize x targetsize]
# # y = tf.constant([1, 2, 3])

# # sess = tf.Session()

# # # idx [qbatchsize x K]
# # _, idx = tf.nn.top_k(x, 2)

# # idx = idx - 1


# # a = tf.gather(y, idx)