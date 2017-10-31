# _*_ coding:utf-8 _*_

import tensorflow as tf
import tensorflow as tf
import numpy as np
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

def main2():
    with tf.variable_scope('foo'):
        v = tf.get_variable('v', [1], initializer=tf.constant_initializer(1.0))

    with tf.variable_scope('too'):
        v = tf.get_variable('v', [1], initializer=tf.constant_initializer(2.0))

    with tf.variable_scope('foo', reuse=True):
        print(v.value)
        v1 = tf.get_variable('v', [1])
        print(v == v1)

    with tf.variable_scope('bar', reuse=None):
        v = tf.get_variable('v', [1])

    print('---------------------------')
    v = tf.get_variable('v', [1])
    v1 = tf.get_variable('v1', [1], initializer=tf.constant_initializer(9.0))
    print(v.name, v1.name)

    with tf.variable_scope('loo'):
        with tf.variable_scope('koo'):
            v2 = tf.get_variable('v', [1], initializer=tf.constant_initializer(3.3))
            print(v2.name)
    with tf.variable_scope('', reuse=True):
        v3 = tf.get_variable('loo/koo/v', [1])
        print(v3.name)
        print(v3 == v)

def main3():
    v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
    v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')
    result = v1 + v2

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        saver.save(sess, 'path/model.ckpt')

def main4():
    v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
    v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')
    result = v1 + v2

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'path/model.ckpt')
        print(sess.run(result))

def main5():
    saver = tf.train.import_meta_graph('path/model.ckpt/model.ckpt.meta')
    with tf.Session() as sess:
        saver.restore(sess, 'path/model.ckpt')
        print(sess.run(tf.get_default_graph().get_tensor_by_name)('add:0'))

if __name__ == '__main__':
    main5()