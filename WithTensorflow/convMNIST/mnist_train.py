# _*_ coding:utf-8 _*_

import tensorflow as tf
import mnist_inference_conv
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 5000
MOVING_AVERAGE_DECAY = 0.99

def train(mnist):
    x = tf.placeholder(tf.float32,
                       [BATCH_SIZE,
                        mnist_inference_conv.IMAGE_SIZE,
                        mnist_inference_conv.IMAGE_SIZE,
                        mnist_inference_conv.NUM_CHANNELS],
                       name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference_conv.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = mnist_inference_conv.inference(x, True, regularizer)

    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.images.shape[0] / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op('train')

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (BATCH_SIZE,
                                          mnist_inference_conv.IMAGE_SIZE,
                                          mnist_inference_conv.IMAGE_SIZE,
                                          mnist_inference_conv.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})

            if i % 500 == 0:
                print('Trainging step(s): %d, loss is: %g' % (step, loss_value))
                # xt, yt = mnist.validation.next_batch(BATCH_SIZE)
                # reshaped_x = np.reshape(xt, (BATCH_SIZE,
                #                              mnist_inference_conv.IMAGE_SIZE,
                #                              mnist_inference_conv.IMAGE_SIZE,
                #                              mnist_inference_conv.NUM_CHANNELS))
                # average_acc = sess.run(accuracy, feed_dict={x: reshaped_x, y_: yt})
                # print('Training step(s) : %d, accuracy is: %g' % (step, average_acc))


def main(argv=None):
    mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()

