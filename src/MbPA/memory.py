import tensorflow as tf 






class Memory():

    def __init__(self, embeddingsize=50, batch_size=128, capacity_multiplier=10, target_size=10):
        self.batch_size = batch_size
        self.capacity = batch_size * capacity_multiplier
        self.embeddingsize = embeddingsize
        self.target_size = target_size
        self.Memory = tf.Variable(tf.zeros(self.capacity, self.embeddingsize), name='KEYS')
        self.Values = tf.Variable(tf.zeros(self.capacity, self.target_size), name='VALUES')

        self.pointer = 0


    def write(hs, values):
        # hs: shape = (batch_size, embeddingsize) 
        # values: shape = (batch_size, target_size)
        # we assume, capacity is a multiple of batch-size!!!!

        indices = tf.Variable(tf.range(start=self.pointer, limit=self.pointer+self.batch_size))
        tf.scatter_update(self.Memory, indices, updates=hs)
        tf.scatter_update(self.Values, indices, updates=values)

        if self.pointer + self.batch_size > self.capacity:
            self.pointer = 0
        else:
            self.pointer += batch_size

