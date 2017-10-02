import tensorflow as tf


W = tf.constant([10, 100], name="W")
x = tf.placeholder(tf.int32, name="x")
b = tf.placeholder(tf.int32, name="b") 

Wx = tf.multiply(W, x, name="Wx")
y = tf.add(Wx, b, name="y")

y_ = tf.subtract(x, b, name="y_")

with tf.Session() as sess:
    print("Intermediate Results: Wx= ", sess.run(Wx, feed_dict={ x:[3,33] } ))
    print("Final Result: y = ", sess.run(y, feed_dict={ x:[5,50], b:[7,9] } ))
    print("Intermediate Results specified: y = ", sess.run(fetches=y, feed_dict={ Wx:[100, 1000], b:[7, 9]}))
    print("Two Results specified: y, y_ = ", sess.run(fetches=[y,y_], feed_dict={ x:[5, 50], b:[7, 9]}))


writer = tf.summary.FileWriter('./m2_example4', sess.graph)
writer.close()
