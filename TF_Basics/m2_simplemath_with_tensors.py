import tensorflow as tf

x = tf.constant([100,200,300], name="x")
y = tf.constant([1,2,3], name="y")

sum_x = tf.reduce_sum(x, name='sum_x')
prod_y = tf.reduce_prod(y, name='prod_y')

final_div = tf.div(sum_x, prod_y, name='final_div')

final_mean = tf.reduce_mean([sum_x, prod_y], name='final_mean')

sesss = tf.Session();

print("x: ", sesss.run(x))
print("y: ", sesss.run(y))
print("sum(x): ", sesss.run(sum_x))
print("prod(y): ", sesss.run(prod_y))
print("sum(x) / prod(y): ", sesss.run(final_div))
print("mean(sum(x) / prod(y)): ", sesss.run(final_mean))


writer = tf.summary.FileWriter('./m2_example4', sesss.graph)

writer.flush()
writer.close()
sesss.close()