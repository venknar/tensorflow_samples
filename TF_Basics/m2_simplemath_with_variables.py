import tensorflow as tf


W = tf.Variable([2.5, 4.0], tf.float32, name="W")
x = tf.placeholder(tf.float32, name="x")
b = tf.Variable([5.0, 10.0], tf.float32, name="b") 

y = W * x + b

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print("Final Result : Wx + b - ", sess.run(y, feed_dict={x:[10, 100]}))

number = tf.Variable(2)
multiplier = tf.Variable(1)

init = tf.global_variables_initializer()

result = number.assign(tf.multiply(number, multiplier))

with tf.Session() as sess:
    sess.run(init)

    for i in range(10):
        print("Result Number * multiplier - ", sess.run(result))
        print("Increment multiplier by one - ", sess.run(multiplier.assign_add(1)))

writer = tf.summary.FileWriter('./m2_example4', sess.graph)
writer.close()
