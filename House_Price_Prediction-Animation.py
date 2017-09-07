import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# generation some house sizes between 1000 and 3500 (typical sq. ft of house)
num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=num_house)
# print(house_size)

# generate the house price based on house sizes with a random noise added.
np.random.seed(42)
house_price = house_size * 100 + np.random.randint(low=20000, high=70000, size=num_house)
# print(house_price)

def normalize(array):
    return ((array - array.mean()) / array.std())

# Let us 70% of the data set as the training set for the algorithm 
num_train_samples = math.floor(0.7 * num_house)

###### Step 1 - Data Preparation - We are defining the training data set and test data set from the given dataset.

# Define the Training data based on the num_train_samples
train_house_size = house_size[:num_train_samples]
train_house_price = house_price[:num_train_samples]

train_house_size_norm = normalize(train_house_size)
train_house_price_norm = normalize(train_house_price)

# Define the Test data 
test_house_size = house_size[num_train_samples:]
test_house_price = house_price[num_train_samples:]

test_house_size_norm = normalize(test_house_size)
test_house_price_norm = normalize(test_house_price)

# Let us define the place holder for the data in the tensor graph.
# These place holders are used to pass the data to the gradient descent algorithm.
tf_house_size = tf.placeholder("float", name="house_size")
tf_house_price = tf.placeholder("float", name="price")

# Let us define the tensor variables which changes as we train the dataset.
# These values are initialized to some random values to start with.
tf_size_factor = tf.Variable(np.random.randn(), name="size_factor")
tf_price_offset = tf.Variable(np.random.randn(), name="price_offset")

# Define the inference functions that predicts the house price based on the house size.
tf_price_pred = tf.add(tf.multiply(tf_size_factor, tf_house_size), tf_price_offset)

# Define the losss function.
tf_cost = tf.reduce_sum(tf.pow(tf_price_pred - tf_house_price, 2)) / (2 * num_train_samples)

# Define the learning rate
learning_rate = 0.1

# Define the optimizer that will minimize the losss defined in the operation.
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

# Initialize the variables.
init = tf.global_variables_initializer()

# There are 4 steps to prepare before we execute the tensor flow graph.
# 1. Prepare the data - In this case it is the test and training dataset.
# 2. Define the inference function - In this case the inference function is the predicate that predicts the price based on the size and size factor and the price offset
# 3. Define the Loss calculation - The losss calculation is defined using the above predicate.
# 4. Finally the Optimizer - This optimizer is what we want to optimize based on the loss function.

# Launch the graph in the session
with tf.Session() as sess:

    sess.run(init)

    display_every = 2
    num_training_iter = 50

    # Calculate the number of lines to animate. 
    fit_num_plots = math.floor(num_training_iter/display_every)
    fit_size_factor = np.zeros(fit_num_plots)
    fit_price_factor = np.zeros(fit_num_plots)
    fit_plot_idx = 0

    # Iterate over the training data.
    for iteration in range(num_training_iter):

        for (x,y) in zip(train_house_size_norm, train_house_price_norm):
            sess.run(optimizer, feed_dict = {tf_house_size: x, tf_house_price: y})

        # Display the current status
        if (iteration + 1) % display_every == 0:
            c = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_house_price: train_house_price_norm})
            print("Iteration #: ", '%04d' % (iteration + 1), "Cost = ", "{:9f}".format(c), \
                "Size Factor = ", sess.run(tf_size_factor), "Price Offset = ", sess.run(tf_price_offset))
            
            # Save the fit_size factor and price_offset to allow animation of learning process.
            fit_size_factor[fit_plot_idx] = sess.run(tf_size_factor)
            fit_price_factor[fit_plot_idx] = sess.run(tf_price_offset)
            fit_plot_idx += 1

    print("Optimisation Finished!")
    training_cost = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_house_price: train_house_price_norm})
    print("Trained Cost = ", training_cost, "Size Factor = ", sess.run(tf_size_factor), "Price Offset =  ", sess.run(tf_price_offset))

    # Plot the graph again.
    plt.rcParams["figure.figsize"] = (10,8)
    plt.figure()
    plt.ylabel("Price")
    plt.xlabel("Size (sq.ft)")
    plt.plot(train_house_size, train_house_price, 'go', label='Training data')
    plt.plot(test_house_size, test_house_price, 'mo', label='Testing data')
    plt.plot(train_house_size_norm * train_house_size.std() + train_house_size.mean(),
             (sess.run(tf_size_factor) * train_house_size_norm + sess.run(tf_price_offset)) * train_house_price.std() + train_house_price.mean(),
             label='Learned Regression')
 
    plt.legend(loc='upper left')
    plt.show()
    

    # 
    # Plot another graph that animation of how Gradient Descent sequentually adjusted size_factor and price_offset to 
    # find the values that returned the "best" fit line.
    fig, ax = plt.subplots()
    line, = ax.plot(house_size, house_price)

    plt.rcParams["figure.figsize"] = (10,8)
    plt.title("Gradient Descent Fitting Regression Line")
    plt.ylabel("Price")
    plt.xlabel("Size (sq.ft)")
    plt.plot(train_house_size, train_house_price, 'go', label='Training data')
    plt.plot(test_house_size, test_house_price, 'mo', label='Testing data')

    def animate(i):
        line.set_xdata(train_house_size_norm * train_house_size.std() + train_house_size.mean())  # update the data
        line.set_ydata((fit_size_factor[i] * train_house_size_norm + fit_price_factor[i]) * train_house_price.std() + train_house_price.mean())  # update the data
        return line,
 
     # Init only required for blitting to give a clean slate.
    def initAnim():
        line.set_ydata(np.zeros(shape=house_price.shape[0])) # set y's to 0
        return line,

    ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, fit_plot_idx), init_func=initAnim,
                                 interval=1000, blit=True)

    plt.show()   
    