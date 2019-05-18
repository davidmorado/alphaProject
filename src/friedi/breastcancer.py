import tensorflow as tf
from sklearn.datasets import load_breast_cancer


X, y = load_breast_cancer(return_X_y=True)
X = X
y = y

# hyperparameter settings
# number of nodes in the two layers
n1 = 15
n2 = 5
learning_rate = 0.1
epochs = 10


# input layer
image = tf.placeholder(dtype=tf.float32, shape=(None, 30), name="image")
label = tf.placeholder(dtype=tf.int64, shape=(None, 1), name="label")

# first hidden layer
w1 = tf.Variable(tf.random_normal([30, n1], stddev=0.35), name="w1")
b1 = tf.Variable(tf.random_normal([n1], stddev=0.35), name="b1")
a1 = tf.matmul(image, w1) + b1
z1 = tf.nn.leaky_relu(a1)
#z1 = tf.math.sigmoid(a1)

# second hidden layer
w2 = tf.Variable(tf.random_normal([n1, n2], stddev=0.35), name="w2")
b2 = tf.Variable(tf.random_normal([n2], stddev=0.35), name="b2")
a2 = tf.matmul(z1, w2) + b2
z2 = tf.nn.leaky_relu(a2)
#z2 = tf.math.sigmoid(a2)

# output layer
w3 = tf.Variable(tf.random_normal([n2, 1], stddev=0.35), name="w3")
b3 = tf.Variable(tf.random_normal([1], stddev=0.35), name="b3")
a3 = tf.matmul(z2, w3) + b3
yhat = tf.math.sigmoid(a3)

# using softmax as cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=yhat, labels=label))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.Session() as sess:

	# initialize variables
	sess.run(tf.global_variables_initializer())

	for epoch in range(epochs):

		for i in range(len(y)):

			# use SGD
			obs = X.T[:, i].reshape(1, 30)
			lab = y[i].reshape(1,1)
			feed_dict = {image: obs, label: lab}

			sess.run(optimizer, feed_dict)
			print(sess.run(lab, feed_dict))
			#yhat = sess.run(yhat, feed_dict)
			#print(yhat)

		print("Loss in Epoch " + str(epoch) + ":")
		print(sess.run(cost, feed_dict))

	writer = tf.summary.FileWriter("breastcancer_logs/", sess.graph)









# import tensorflow as tf
# from sklearn.datasets import load_breast_cancer


# X, y = load_breast_cancer(return_X_y=True)

# # hyperparameter settings
# # number of nodes in the two layers
# n1 = 15
# n2 = 5
# learning_rate = 1e-6
# epochs = 10


# # input layer
# image = tf.placeholder(dtype=tf.float32, shape=(None, 30), name="image")
# label = tf.placeholder(dtype=tf.int64, shape=(None, 1), name="label")

# # first hidden layer
# w1 = tf.Variable(tf.truncated_normal([30, n1], mean=0, stddev=0.08), name="w1")
# b1 = tf.Variable(tf.truncated_normal([n1], mean=0, stddev=0.08), name="b1")
# a1 = tf.add(tf.matmul(image, w1), b1)
# z1 = tf.nn.leaky_relu(a1)
# #z1 = tf.math.sigmoid(a1)


# # second hidden layer
# w2 = tf.Variable(tf.truncated_normal([n1, n2], mean=0, stddev=0.08), name="w2")
# b2 = tf.Variable(tf.truncated_normal([n2], mean=0, stddev=0.08), name="b2")
# a2 = tf.add(tf.matmul(z1, w2), b2)
# z2 = tf.nn.leaky_relu(a2)
# #z2 = tf.math.sigmoid(a2)

# # output layer
# w3 = tf.Variable(tf.truncated_normal([n2, 1], mean=0, stddev=0.08), name="w3")
# b3 = tf.Variable(tf.truncated_normal([1], mean=0, stddev=0.08), name="b3")
# a3 = tf.add(tf.matmul(z2, w3), b3)
# yhat = tf.math.sigmoid(a3)

# # using softmax as cost function
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yhat, labels=label))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# with tf.Session() as sess:

# 	# initialize variables
# 	sess.run(tf.global_variables_initializer())

# 	for epoch in range(epochs):

# 		for i in range(len(y)):

# 			obs = X.T[:, i].reshape(1, 30)
# 			lab = y[i].reshape(1,1)

# 			feed_dict = {image: obs, label: lab}
# 			sess.run(optimizer, feed_dict)

# 		print("Loss in Epoch " + str(epoch) + ":")
# 		print(sess.run(cost, feed_dict))

# 	writer = tf.summary.FileWriter("breastcancer_logs/", sess.graph)

























