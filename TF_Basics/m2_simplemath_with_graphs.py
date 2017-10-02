import tensorflow as tf

g1 = tf.Graph()
g2 = tf.Graph()

with g1.as_default():
    with tf.Session() as sess:
        A = tf.constant([5,7], tf.int32, name="A")
        x = tf.placeholder(tf.int32, name="x")
        b = tf.constant([3,4], tf.int32, name="b")

        y = A * x + b
        print(sess.run(y, feed_dict={x:[10, 100]}))

        assert y.graph is g1

with g2.as_default():
    with tf.Session() as sess:
        A = tf.constant([6,8], tf.int32, name="A")
        x = tf.placeholder(tf.int32, name="x")
        b = tf.constant([4,5], tf.int32, name="b")

        y = A * x + b
        print(sess.run(y, feed_dict={x:[10, 100]}))
        
        assert y.graph is g2


