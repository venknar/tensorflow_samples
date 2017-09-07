import tensorflow as tf

sess = tf.Session()

hello = tf.constant("Hello Pluralsight from Tensor Flow")
print(sess.run(hello))


a = tf.constant(20)
b = tf.constant(3)
print('a + b = {0}'.format(sess.run(a+b)))