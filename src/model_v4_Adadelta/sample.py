import numpy as np

def sample(x, y, tr, n_classes):

    sample_x = []
    sample_y = []
    
    for category in range(n_classes):
        idx_category = [idx for idx in range(y.shape[0]) if  y[idx, category] == 1]
        x_tmp = x[idx_category]
        y_tmp = y[idx_category]
        n = int(x_tmp.shape[0] * tr)

        sample_x.append(x_tmp[:n])
        sample_y.append(y_tmp[:n])

    sample_x = np.concatenate(sample_x, axis=0)
    sample_y = np.concatenate(sample_y, axis=0)
    return (sample_x, sample_y)