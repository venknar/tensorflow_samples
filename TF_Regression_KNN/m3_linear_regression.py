import tensorflow as tf

# Model Parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

# Model Input and output
x = tf.placeholder(tf.float32)
# Linear Model is our predicton and tuning.
linear_model = W * x + b

y = tf.placeholder(tf.float32)

# Loss
loss = tf.reduce_sum(tf.square(linear_model - y))

#Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)

# We want the optimizer to minimimze the loss 
train = optimizer.minimize(loss)

# training data
x_data = [1, 2, 3, 4]
y_data = [0, -1, -2, -3]

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(1000):
        sess.run(train, {x : x_data, y : y_data})
    
    # Evaluate the training accuracy
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_data, y: y_data})

    print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
