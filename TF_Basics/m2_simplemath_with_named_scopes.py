import tensorflow as tf

A = tf.constant([4], tf.int32, name="A")
B = tf.constant([5], tf.int32, name="B")
C = tf.constant([6], tf.int32, name="C")

x = tf.placeholder(tf.int32, name="x")

with tf.name_scope("Equation_1"):
    Ax2_1 = tf.multiply(A, tf.pow(x,2), name="Ax2_1")
    Bx = tf.multiply(B, x, name="Bx")
    y1 = tf.add_n([Ax2_1, Bx, C], name="y1")

with tf.name_scope("Equation_2"):
    Ax2_2 = tf.multiply(A, tf.pow(x,2), name="Ax2_2")
    Bx2 = tf.multiply(B, x, name="Bx2")
    y2 = tf.add_n([Ax2_2, Bx2], name="y1")

with tf.name_scope("Final_Scope"):
    y = y1 + y2

with tf.Session() as sess:
    print(sess.run(y, feed_dict={x:[10]}))

writer = tf.summary.FileWriter('./m2_example4', sess.graph)
writer.close()
