import tensorflow as tf

import matplotlib.image as mp_img
import matplotlib.pyplot as pyplot
import os

filename = "./DandelionFlower.jpg"
filename1 = './TaraxacumOfficinaleSeed.jpg'

image = mp_img.imread(filename1)

print("Image Shape ", image.shape)
print("Image Array ", image)

pyplot.imshow(image)
pyplot.show()

x = tf.Variable(image, name='x')

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # Transpose the image x and y 
    transpose = tf.image.transpose_image(x)
    result = sess.run(transpose)

    print("Transposed Image Shape ", result.shape)
    pyplot.imshow(result)
    pyplot.show()
