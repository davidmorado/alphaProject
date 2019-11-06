import numpy as np

def getBatchIndices(x_train, batchSize=32):
        idx = np.random.randint(0, x_train.shape[0], x_train.shape[0])
        num_batches = int(x_train.shape[0]/batchSize)
        idxs = np.array_split(idx, num_batches)
        return idxs
