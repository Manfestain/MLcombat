# _*_ coding:utf-8 _*_

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

HIDDEN_SIZE = 30
NUM_LAYERS = 2
TIME_STEPS = 10
TRAINING_STEPS = 10000
BATCH_SIZE = 32
INPUT_SIZE = 1
OUTPUT_SIZE = 1

TRAINING_EXAMPLES = 10000
TESTING_EXAMPLES = 1000
SAMPLE_GAP = 0.01
LEARNING_RATE = 0.006

# 获得数据
test_start = TRAINING_EXAMPLES * SAMPLE_GAP
test_end = (TRAINING_EXAMPLES + TESTING_EXAMPLES) * SAMPLE_GAP
data = np.sin(np.linspace(0, test_start, TESTING_EXAMPLES, dtype=np.float32))
normalize_data = (data - np.mean(data)) / np.std(data)
normalize_data = normalize_data[:, np.newaxis]
# print(normalize_data.shape)
# plt.figure()
# plt.plot(normalize_data)
# plt.show()

# 形成训练集
train_x, train_y = [], []
for i in range(len(normalize_data) - TIME_STEPS - 1):
    x = normalize_data[i: i + TIME_STEPS]
    y = normalize_data[i + 1: i + TIME_STEPS + 1]
    train_x.append(x.tolist())
    train_y.append(y.tolist())
# print('x: ', np.array(train_x, dtype=np.float32).shape)
# print('y: ', np.array(train_y, dtype=np.float32).shape)
train_x = np.array(train_x, dtype=np.float32)
train_y = np.array(train_y, dtype=np.float32)

# 定义神经网络变量
X = tf.placeholder(tf.float32, [None, TIME_STEPS, INPUT_SIZE])
Y = tf.placeholder(tf.float32, [None, TIME_STEPS, OUTPUT_SIZE])
weights = {
    'in': tf.Variable(tf.random_normal([INPUT_SIZE, HIDDEN_SIZE])),
    'out': tf.Variable(tf.random_normal([HIDDEN_SIZE, 1]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[HIDDEN_SIZE, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}

# 定义LSTM网络
def lstm(batch):
    w_in = weights['in']
    b_in = biases['in']
    inputs = tf.reshape(X, [-1, INPUT_SIZE])
    inputs_rnn = tf.matmul(inputs, w_in) + b_in
    inputs_rnn = tf.reshape(inputs_rnn, [-1, TIME_STEPS, HIDDEN_SIZE])

    cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * NUM_LAYERS)

    init_state = cell.zero_state(batch_size=batch, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, inputs_rnn, initial_state=init_state, dtype=tf.float32)

    output = tf.reshape(output_rnn, [-1, HIDDEN_SIZE])
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out

    return pred, final_states

# 训练模型
def train_lstm():
    pred, _ = lstm(BATCH_SIZE)
    # plot_pred = np.asarray(tf.reshape(pred, [320]), dtype=np.float32)
    # plt.figure()
    # plt.plot(plot_pred)
    # plt.show()
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(TRAINING_STEPS):
            step = 0
            start = 0
            end = start + BATCH_SIZE
            while(end < len(train_x)):
                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[start: end], Y: train_y[start: end]})
                start += BATCH_SIZE
                end = start + BATCH_SIZE
            if i % 1000 == 0:
                print('%d steps, loss is: %f' % (i, loss_))


train_lstm()