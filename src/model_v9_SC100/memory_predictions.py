import keras
from keras import Model
import numpy as np
import pdb

def memory_predictions(model, x_train, x_test, y_train, y_test, num_categories, cache_fraction, theta, lmbd):

    # Specify memory
    mem = Model(inputs=model.get_input_at(0), outputs=model.layers[17].output)

    # subsample data points for memory module (further subsampling from potentially subsampled training data)
    idx = np.random.permutation(np.arange(x_train.shape[0]))[:int(x_train.shape[0]*cache_fraction)]

    x_train_in_mem = x_train[idx, :, :, :]
    y_train_in_mem = y_train[idx, :]

    # Memory keys
    mem_keys = mem.predict(x_train_in_mem)

    # Memory values
    mem_vals = y_train_in_mem

    # Pass items thru memory
    test_mem = mem.predict(x_test)

    # Normalize keys and query
    query = test_mem / np.sqrt( np.tile(np.sum(test_mem**2, axis=1, keepdims=1),(1,test_mem.shape[1])) )
    key = mem_keys / np.sqrt( np.tile(np.sum(mem_keys**2, axis=1, keepdims=1),(1,mem_keys.shape[1])) )

    similarities = np.exp(theta * np.dot(query, key.T))
    p_mem = np.matmul(similarities, mem_vals)
    p_mem = p_mem / np.repeat(np.sum(p_mem, axis=1, keepdims=True), num_categories, axis=1)
    p_model = model.predict(x_test)
    p_combined = (1.0-lmbd) * p_model + lmbd * p_mem

    pred_mem = np.argmax(p_mem, axis=1)
    pred_combined = np.argmax(p_combined, axis=1)
    y_test_int = np.argmax(y_test, axis=1)
    
    mem_acc_test = np.mean(pred_mem==y_test_int)
    comb_acc_test = np.mean(pred_combined==y_test_int)

    return mem_acc_test, comb_acc_test