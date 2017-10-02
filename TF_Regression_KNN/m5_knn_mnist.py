import numpy as np
import tensorflow as tf

# Import MNIST Data
from tensorflow.examples.tutorials.mnist import input_data

# Store the MNIST data into the tmp data
mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

# Using the subset of the data for ensuring that we can run faster. 
training_digits, training_labels = mnist.train.next_batch(5000)
test_digits, test_labels = mnist.test.next_batch(200)

training_digits_pl = tf.placeholder("float", [None, 784])
test_digits_pl = tf.placeholder("float", [784])

# Nearest neighbout calculation using the L1 distance
l1_distance = tf.abs(tf.add(training_digits_pl, tf.negative(test_digits_pl)))

distance = tf.reduce_sum(l1_distance, axis=1)

# Prediction - Get min distance index ()
pred = tf.arg_min(distance, 0)

accuracy = 0

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(len(test_digits)):
        # Get NN neightbour
        nn_index = sess.run(pred, \
        feed_dict={training_digits_pl:training_digits, test_digits_pl:test_digits[i,:]})

        print("Test ", i, "Prediction ", np.argmax(training_labels[nn_index]), "True Label ", np.argmax(test_labels[i]))

        # Calculate the accuracy
        if np.argmax(training_labels[nn_index]) == np.argmax(test_labels[i]):
            accuracy += 1./len(test_digits)


print("Done !")
print("Accuracy : ", accuracy)