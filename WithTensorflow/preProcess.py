

import tensorflow as tf
import matplotlib.pyplot as plt

image_raw_data = tf.gfile.GFile("E:/LoveWallpaper/295547-106.jpg", 'rb').read()

def demo():
    with tf.Session() as sess:
        img_data = tf.image.decode_jpeg(image_raw_data)
        print(img_data.eval())

        plt.imshow(img_data.eval())
        plt.show()

        img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)

def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    else:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    return tf.clip_by_valuer(image, 0.0, 1.0)

def preprocess_for_train(image, height, width, bbox):
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])

    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    bbox_begin
