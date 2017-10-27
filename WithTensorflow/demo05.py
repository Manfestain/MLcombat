# _*_ coding:utf-8 _*_

import tensorflow as tf
from numpy.random import RandomState

def get_weight(shape, a):
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(a)(var))
    tf.add_to_collection('weight', var)
    return var

def main():
    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))

    batch_size = 8
    test = tf.constant([2.0, 3.0])

    layer_dimension = [2, 4, 8, 4, 1]
    n_layers = len(layer_dimension)
    cur_layer = x
    in_dimension = layer_dimension[0]

    #   使用循环生成5层全连接的神经网络模型
    for i in range(1, n_layers):
        out_dimension = layer_dimension[i]
        weight = get_weight([in_dimension, out_dimension], 0.001)
        bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
        cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
        in_dimension = layer_dimension[i]

    #   生成最终的损失函数
    mes_loss = tf.reduce_mean(tf.square(y_ - cur_layer))
    tf.add_to_collection('losses', mes_loss)
    loss = tf.add_n(tf.get_collection('losses'))

    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
    data_size = 500
    rdm = RandomState(1)
    X = rdm.rand(data_size, 2)
    Y = [[3 * x1 + 5 * x2] for (x1, x2) in X]

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(5000):
            start = (i * batch_size) % data_size
            end = min(start + batch_size, data_size)
            sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        for i in range(len(tf.get_collection('weight'))):
            cur_value = tf.matmul(test, tf.get_collection('weight')[i])
            tf.assign(test, cur_value, validate_shape=False)
        print(sess.run(cur_value))
if __name__ == '__main__':
    main()
