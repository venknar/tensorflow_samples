import tensorflow as tf
from PIL import Image
import matplotlib.image as mp_img
import matplotlib.pyplot as pyplot
import os

original_images_list = ["./DandelionFlower.jpg", "./TaraxacumOfficinaleSeed.jpg"]

# Make a queue of all the frimage files. 
filename_queue = tf.train.string_input_producer(original_images_list)

# Read an entire file
image_reader = tf.WholeFileReader()

with tf.Session() as sess:
    # Coordinator the loading of all the files
    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    images_list = []

    for i in range(len(original_images_list)):
        # Read a whole file from the queue, the first returned values is the file name and second parameter is the image.
        _, image_file = image_reader.read(filename_queue)

        # Decode the image as jpeg file        
        image = tf.image.decode_jpeg(image_file)

        # Get the resized images
        image = tf.image.resize_images(image, [224,224])

        image.set_shape((224,224,3))

        image = tf.image.flip_up_down(image)
        image = tf.image.central_crop(image, central_fraction=0.5)
        
        # Get the image tensor after resize
        image_resized = sess.run(image)
        print(image_resized.shape)

        # Converts from numpy to the tensor image.
        image_tensor = tf.stack(image_resized)            

        # Expand dims to add a new dimension
        print(image_tensor)
        images_list.append(image_tensor)

    # Finsish all the coordinator. 
    coord.request_stop()
    coord.join(threads)

    images_tensor = tf.stack(images_list)
    print(images_tensor)

    writer = tf.summary.FileWriter('./m4_example4', sess.graph)

    summary_str = sess.run(tf.summary.image("images", images_tensor, max_outputs=4))
    writer.add_summary(summary_str)

    writer.close()
        