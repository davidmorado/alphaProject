
import tensorflow as tf

x_train, x_val, x_test, y_train, y_val, y_test = get_dataset('cifar100', normalize=True)
num_categories = y_train.shape[1]
input_shape = x_train.shape[1:]



# firststage densenet

x = tf.placeholder(tf.float32, shape=(?, 32, 32, 3))
y = tf.placeholder(tf.int32, shape=(?, num_categories))

