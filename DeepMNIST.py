# DeepMNIST.py
# DeepMNIST to classify handwritten digits from MNIST dataset

import tensorflow as tf
import time

# Importing the MNIST data from this site - https://www.tensorflow.org/get_started/mnist/beginners
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

logPath = "./tb_logs/"

with tf.name_scope("MNIST_Input"):
    # Let us define the placeholder for the MNIST data in the tensor flow graph.
    x = tf.placeholder(tf.float32, shape=[None, 28 * 28])
    # y_ is called y bar and is a 10 element vector, containing the probability of each digit 
    # (0-9) class. Such as [0.14, 0.8, 0,0,0,0,0, 0.06] 
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

with tf.name_scope("Inputy_Reshape"):
    # Change the MNIST image data from a list of values to a 28 pixel x 28 pixel x 1 pixel grayscale cube which the Convolution NN can use.
    x_image = tf.reshape(x, [-1,28,28,1], name="x_image")
    tf.summary.image('input_img', x_image, 5)

# Define the helper functions to create the weights and the biases . These needs to be initialized to some non zero value.
def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

# Convolution and Pooling - we do convolution and then pooling to control overfitting.
def conv2d(x, W, name=None):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME", name=name)

def max_pool_2x2(x, name=None):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME", name=name)

def variables_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean", mean)
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar("stddev", stddev)
        tf.summary.scalar("max", tf.reduce_max(var))
        tf.summary.scalar("min", tf.reduce_min(var))
        tf.summary.histogram("histogram", var)
        
with tf.name_scope("Conv1"):
    # Define the layers in the NN.
    # 1st Convolution Layer
    # 32 features of 5 x 5 patch of the image.
    with tf.name_scope("weight"):
        W_conv1 = weight_variable([5,5,1,32], name="weight")
        variables_summaries(W_conv1)

    with tf.name_scope("biases"):
        b_conv1 = bias_variable([32], name="bias")
        variables_summaries(b_conv1)

    # Do the convolution of the images, add bias and push through relu
    conv1_wx_b = conv2d(x_image, W_conv1, name="conv2d") + b_conv1
    tf.summary.histogram('conv1_wx_b', conv1_wx_b)
    h_conv1 = tf.nn.relu(conv1_wx_b, name="relu")
    tf.summary.histogram('h_conv1', h_conv1)

    # Take the results and send through the pool.
    h_pool1 = max_pool_2x2(h_conv1, name="pool")

with tf.name_scope("Conv2"):
    # 2nd convolution Layer
    # Process the 32 features from the convolution layer 1 5 x 5 pathc and return 64 features and bias.
    with tf.name_scope("weight"):
        W_conv2 = weight_variable([5,5,32,64], name="weight")
        variables_summaries(W_conv2)

    with tf.name_scope("biases"):
        b_conv2 = bias_variable([64], name="bias")
        variables_summaries(b_conv2)

    # Do the convolution of the second layer with result from first layer
    conv2_wx_b = conv2d(h_pool1, W_conv2, name="conv2d") + b_conv2
    tf.summary.histogram('conv2_wx_b', conv2_wx_b)

    h_conv2 = tf.nn.relu(conv2_wx_b, name="relu")
    tf.summary.histogram('h_conv2', h_conv2)

    h_pool2 = max_pool_2x2(h_conv2, name="pool")

with tf.name_scope("FC"):
    # Fully connected layer
    W_fc1 = weight_variable([7 * 7 * 64,1024], name="weight")
    b_fc1 = bias_variable([1024], name="bias")

    # connect the output of the 2nd layer to the fully connected layer
    h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name="relu")

# Drop out to remove some of the neurons to avoid overfitting of the data.
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope("RO"):
    # Readout layer
    W_fc2 = weight_variable([1024,10], name="weight")
    b_fc2 = bias_variable([10], name="bias")

# define the final model
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

with tf.name_scope("cross_entropy"):
    # Define the loss function - loss is the cross entropy.
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

learning_rate = 0.5

with tf.name_scope("loss_optimizer"):
    # Define the optimizer that will minimize the losss defined in the operation.
    # We want to minimize the cross entropy in each step.
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope("accuracy"):
    # Evaluate how well the model did. compare the digit in the predicted (y_) with the actual (y) 
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar("cross_entropy_scl", cross_entropy)
tf.summary.scalar("training_accuracy", accuracy)

# Merge summaries
summarize_all = tf.summary.merge_all()

# Using Interactive session makes it the default sessions so we do not need to pass sess
sess = tf.InteractiveSession()

# Perform the initialization of the session with the given global variables.
sess.run(tf.global_variables_initializer())

# TB - Write the graph out so that we can see its structure.
tbWriter = tf.summary.FileWriter(logPath, sess.graph)

start_time = time.time()
end_time = time.time()

num_steps = 500
display_every = 100

for i in range(num_steps):
    batch = mnist.train.next_batch(50)
    _, summary = sess.run([optimizer, summarize_all], feed_dict = {x: batch[0], y_: batch[1], keep_prob:0.5})

    if i % display_every == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob:1.0})
        end_time = time.time()
        print("Step {0}, Elapsed Time {1:.2f} seconds, Training Accuracy {2:.3f}%".format(i, end_time-start_time, train_accuracy * 100))
        # Write summary to log
        tbWriter.add_summary(summary, i)

    # Display summary
    end_time = time.time()
    #print("Total Training time for {0} batches: {1:.2f} seconds.".format(i+1, end_time-start_time))

# Test the accuracy on the test data.
print("Test Accuracy - {0:.3f}%".format(accuracy.eval(feed_dict={x:mnist.test.images, y_: mnist.test.labels, keep_prob:1.0})* 100))

sess.close()

