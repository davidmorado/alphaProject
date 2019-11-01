import tensorflow as tf
from tqdm import tqdm
import numpy as np

from utils import getBatchIndices
from models import Stage1, Stage2
from data_loader import get_dataset
from memory import memory
DEBUG_MODE = True

# get training data
x_train, x_val, x_test, y_train, y_val, y_test = get_dataset('cifar10', normalize=True)
num_categories = y_train.shape[1]
N,h,w,c = x_train.shape

# reduce data for debugging
if DEBUG_MODE: 
    x_train, y_train = x_train[:100, :, :, :], y_train[:100, :]
    x_val, y_val = x_val[:2, :, :, :], y_val[:2, :]
    x_test, y_test = x_test[:2, :, :, :], y_test[:2, :]

# hyperparameter
batch_size = 10
train_epochs = 2
infer_epochs = 2
learning_rate = 0.001
embedding_size = 100
memory_capacity = 155
query_size = 50

# Stage1 
memory = memory(embedding_size=embedding_size, capacity=memory_capacity, target_size=num_categories, K=query_size)
x = tf.placeholder(tf.float32, shape=(None, h, w, c), name='x')
y = tf.placeholder(tf.float32, shape=(None, num_categories), name='y')
embeddings = Stage1(x)

# Stage2
memory.write(embeddings, y)
predictions = Stage2(embeddings) # play around with tf.AUTO_REUSE
cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=predictions, onehot_labels=y), name='cost')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam').minimize(cost)
correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1), name='correct_predictions')
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='train_accuracy')

# Stage2 for augmented prediction
hit_keys, hit_values, weights = memory.read(embeddings)
predictions_aug = Stage2(hit_keys, reuse=True) 
cost_aug = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=predictions_aug, onehot_labels=hit_values, weights=weights), name='cost_aug')
optimizer_aug = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam_aug').minimize(cost_aug, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Stage2'))
augmented_prediction = Stage2(embeddings, reuse=True)
IsPredictionCorrect = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1), name='IsPredictionCorrect')



with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    print('Training...')
    for epoch in range(train_epochs):
        batch_idxs = getBatchIndices(x_train, batch_size)
        for bidx in tqdm(batch_idxs):
            sess.run(optimizer, feed_dict = {x: x_train[bidx], y: y_train[bidx]})
        # train_loss = sess.run(cost, feed_dict = {x: x_train[:10000], y: y_train[:10000]})
        # print('train loss:\t {}'.format(train_loss))
    tf.summary.FileWriter('tmp/graphs', sess.graph)
    print('Graph written to tmp/graphs')
    save_path = saver.save(sess, "tmp/models/model.ckpt")
    print("Model written to: %s" % save_path)

    # inference on one data point from x_val
    #loader = tf.train.import_meta_graph('tmp/models/model.ckpt.meta')
    correctSumAdapted = 0
    total = len(x_val)
    print('Inferring...')
    for img, label in zip(x_val, y_val):
        img = np.expand_dims(img, axis=0)
        label = np.expand_dims(label, axis=0)
        print('Adapting second Stage Parameters...')
        for epoch in range(infer_epochs):
            sess.run(optimizer_aug, feed_dict={x: img, y: label})
        correctSumAdapted += sess.run(IsPredictionCorrect)
        print('Prediction made. Restoring 2nd Stage parameters.')
        saver.restore(sess, 'tmp/models/model.ckpt.meta')
        print('Parameters restored.')
    


# inference
# tf.reset_default_graph()
# with tf.Session() as sess:
#     for unseen_x in x_val:
#         tf.train.Saver().restore(sess, "tmp/models/model.ckpt")
#         for epoch in range(infer_epochs):

# something like this:
# sess.run(optimizer, feed_dict = {x: x_train[bidx], y: y_train[bidx]}, trainable_vars = tf.get_collection('Stage2'))

