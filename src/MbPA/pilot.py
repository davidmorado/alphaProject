import tensorflow

from model import conv_net, secondStage
from memory import Memory

from keras.datasets import cifar10
from keras.utils import to_categorical
import numpy as np

# Data Loading and Preprocessing
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
input_shape = x_train.shape[1:]
num_classes = np.max(y_test)+1
num_samples = x_train.shape[0]

x_train = x_train/255
x_test = x_test/255
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

print(x_train.shape)
print(y_train.shape)





# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')
y = tf.placeholder(tf.float32, shape=(None, 10), name='output_y')




epochs = 10
batch_size = 128
learning_rate = 0.001


M = Memory(secondStage)
embeddings = conv_net(x)
logits = M.model(embeddings)



# Loss and Optimizer
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')


def training_step(session, optimizer, batch_features, batch_labels):
    _, h = session.run([optimizer, embeddings],
                feed_dict={
                    x: batch_features,
                    y: batch_labels
                })
    return h

def print_stats(session, batch_features, batch_labels, cost, accuracy):
    loss = sess.run(cost, 
                    feed_dict={
                        x: batch_features,
                        y: batch_labels
                    })
    valid_acc = sess.run(accuracy, 
                         feed_dict={
                             x: valid_features,
                             y: valid_labels
                         })
    
    print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(loss, valid_acc))

with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in load_preprocess_training_batch(batch_i, batch_size):
                _, hs = training_step(sess, optimizer, keep_probability, batch_features, batch_labels)
                M.write(hs, batch_labels)
                
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, accuracy)


