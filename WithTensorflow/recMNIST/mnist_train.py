# _*_ coding：utf-8 _*_

import os
import tensorflow as tf
import mnist_inference
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 100

# 衰减的学习率
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DEACY = 0.99

# 正则化损失
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 3000
MOVING_AVERAGE_DECAY = 0.99

# 定义保存的路径和文件名
MODEL_SAVE_PATH = "/path/to/"
MODEL_NAME = "model.ckpt"

def train(mnist):
    # 定义输入输出的占位
    x = tf.placeholder(tf.float32, shape=[None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, shape=[None, mnist_inference.OUTPUT_NODE], name='y-input')

    # 正则化损失，得到前向传播结果
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = mnist_inference.inference(x, regularizer)

    # 定义滑动平均模型
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 定义交叉损失
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # 衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.images.shape[0] / BATCH_SIZE,
        LEARNING_RATE_DEACY
    )

    # 定义优化的方法
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # 训练并保存训练的结果
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})

            if i % 500 == 0:
                print("After %d training step(s), loss on training batch is %g" % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main(argv=None):
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()