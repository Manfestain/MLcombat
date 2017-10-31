# _*_ coding:utf-8 _*_

import tensorflow as tf

# 定义神经网络的神经元数目
INPUT_NODE = 784
LAYER1_NODE = 500
OUTPUT_NODE = 10

# 使用变量管理的方式定义神经网络结构和前向传播算法
# 当程序使用定义好的网络时， 直接调用inference(new_x, True)即可
def inference(input_tensor, reuse=False):

    with tf.variavle_scope('layer1', reuse=reuse):
        weights = tf.get_variable('weights', [INPUT_NODE, LAYER1_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variavle('biases', [LAYER1_NODE],
                                 initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2', reuse=reuse):
        weights = tf.get_variable('weights', [LAYER1_NODE, OUTPUT_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [OUTPUT_NODE],
                                 initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2
