import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 

np.random.seed(101) 
tf.set_random_seed(101) 

# Genrating random linear data 
# There will be 50 data points ranging from 0 to 50 
x = np.linspace(-25, 25, 100) 
y = np.linspace(-25, 25, 100)
y = y**2

# Adding noise to the random linear data 
x += np.random.uniform(-1, 1, 100) 
y += np.random.uniform(-1, 1, 100) 

n = len(x) # Number of data points 

# Plot of Training Data 
"""
plt.scatter(x, y) 
plt.xlabel('x') 
plt.ylabel('y') 
plt.title("Training Data") 
plt.show() 
"""

x_ = tf.placeholder("float")
X = tf.placeholder("float") 
Y = tf.placeholder("float")


W = tf.Variable(np.random.randn(), name = "W") 
b = tf.Variable(np.random.randn(), name = "b") 


learning_rate = 0.001
training_epochs = 50

# getting case weights for a given X for all x
bandwidth = tf.constant(1, dtype="float32")
w = tf.exp(-tf.divide(tf.square(x_ - X), tf.square(tf.multiply(2.0, bandwidth))))

# making the prediction
y_ = tf.reduce_sum(tf.add(tf.multiply(X, W), b))

# Mean Squared Error Cost Function 
cost = tf.reduce_sum(tf.multiply(tf.pow(y_ - Y, 2), w))
  
# Gradient Descent Optimizer 
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Hypothesis function
y_pred = tf.add(tf.multiply(X, W), b)

# getting case weights for a given X for all x
#bandwidth = tf.constant(0.5, dtype="float32")
#w = tf.exp(-tf.divide(tf.square(x_ - X), tf.square(tf.multiply(2.0, bandwidth))))

# making the prediction
#y_ = tf.reduce_sum(tf.multiply(tf.add(tf.multiply(X, W), b), w))

# list for predictions
predictions = []
beta0 = []
beta1 = []

# looping over unseen data points
x_unseen = [-25, -20, -15, -10, 5, 0, 5, 10, 15, 20, 25]
for x_new in x_unseen:

	# Global Variables Initializer 
	init = tf.global_variables_initializer() 

	# Starting the Tensorflow Session 
	with tf.Session() as sess: 
		
		# Initializing the Variables 
		sess.run(init) 
		
		# Iterating through all the epochs 
		for epoch in range(training_epochs): 
			
			# Feeding each data point into the optimizer using Feed Dictionary 
			for (_x, _y) in zip(x, y): 
				sess.run(optimizer, feed_dict = {x_: x_new, X: _x, Y: _y}) 
			
			# Displaying the result after every 50 epochs 
			if (epoch + 1) % 50 == 0: 
				# Calculating the cost a every epoch 
				c = sess.run(cost, feed_dict = {x_: x_new, X : x, Y : y}) 
				print("Epoch", (epoch + 1), ": cost =", c, "W =", sess.run(W), "b =", sess.run(b)) 
		
		# Storing necessary values to be used outside the Session 
		training_cost = sess.run(cost, feed_dict ={x_: x_new, X: x, Y: y}) 
		weight = sess.run(W) 
		beta1.append(W)
		bias = sess.run(b) 
		beta0.append(b)

		# Calculating the prediction for the unseen point
		feed_dict = {x_: x_new, X: x_new}
		yhat = sess.run(y_, feed_dict) 
		predictions.append(yhat)

	#print("Training cost =", training_cost, "Weight =", weight, "bias =", bias, '\n')

print(predictions)

# Plotting the Results

plt.plot(x, y, 'ro', label ='Original data') 
plt.scatter(x_unseen, predictions, label ='Predictions', c='blue', marker='x')

plt.title('Locally Linear Regression Result') 
plt.legend() 
plt.show()


