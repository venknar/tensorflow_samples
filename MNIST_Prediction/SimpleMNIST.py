# SimpleMNIST.py
# Simple MN to classify handwritten digits from MNIST dataset

import tensorflow as tf

# Importing the MNIST data from this site - https://www.tensorflow.org/get_started/mnist/beginners
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

# Let us define the placeholder for the MNIST data in the tensor flow graph.

x = tf.placeholder(tf.float32, shape=[None, 28 * 28])

# y_ is called y bar and is a 10 element vector, containing the probability of each digit 
# (0-9) class. Such as [0.14, 0.8, 0,0,0,0,0, 0.06] 
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Define the weights and biases 
W = tf.Variable(tf.zeros([28 * 28, 10]))
b = tf.Variable(tf.zeros([10]))

# Define the prediction or the model.
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Define the loss function - loss is the cross entropy.
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

learning_rate = 0.5

# Define the optimizer that will minimize the losss defined in the operation.
# We want to minimize the cross entropy in each step.
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# Initialize the variables.
init = tf.global_variables_initializer()

# There are 4 steps to prepare before we execute the tensor flow graph.
# 1. Prepare the data - In this case it is the test and training dataset.
# 2. Define the inference function - In this case the inference function is the predicate that predicts the price based on the size and size factor and the price offset
# 3. Define the Loss calculation - The losss calculation is defined using the above predicate.
# 4. Finally the Optimizer - This optimizer is what we want to optimize based on the loss function.

with tf.Session() as sess:
    # Perform the initialization of the session with the given global variables.
    sess.run(init)

    for iter in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)  # get the next random 100 images
        # Let us do the optimization with the given dataset.
        sess.run(optimizer, feed_dict = {x: batch_xs, y_: batch_ys})
    
    # Evaluate how well the model did. compare the digit in the predicted (y_) with the actual (y) 
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # Evaluate the accuracy with the test data. Let us run it over the test data.
    print("\n\n")
    test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print("Test Accuracy: {0}%".format(test_accuracy * 100.0))