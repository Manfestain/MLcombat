# _*_ coding:utf-8 _*_

import tensorflow as tf
from numpy.random import RandomState

def loadData():
    X = np.array([[1.0, 3], [4, 6], [7, 9], [0, 4.0], [4.0, 9], [9, 8], [7, 7]])
    y = np.array([[x1 + x2] for (x1, x2) in X])
    return X, y

def main():
    rdm = RandomState(1)
    X = rdm.rand(100, 2)
    Y = [[x1 + 2 * x2] for (x1, x2) in X]

    x = tf.placeholder(tf.float32, shape=[None, 2], name='x-input')
    y = tf.placeholder(tf.float32, shape=[None, 1], name='y-input')

    w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))

    y_ = tf.matmul(x, w1)

    loss = tf.reduce_mean(tf.square(y_ - y)) + tf.contrib.layers.l1_regularizer(0.4)(w1)
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(1000):
            sess.run(train_step, feed_dict={x: X, y: Y})
        print(sess.run(w1))


if __name__ == '__main__':
    main()