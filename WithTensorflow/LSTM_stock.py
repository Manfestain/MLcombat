# _*_ coding:utf-8 _*_

import pandas as pd
import numpy as np
import tensorflow as tf

HIDDEN_SIZE = 10
INPUT_SIZE = 7
OUTPUT_SIZE = 1
LEARNING_RATE = 0.0006

# 导入数据
f = open('F:/Download/DataSets/stock_dataset/dataset_2.csv')
df = pd.read_csv(f)
data = df.iloc[:, 2:10].values

# 获取训练集
def get_train_data(batch_size=60, time_step=20, train_begin=0, train_end=5800):
    batch_index = []
    data_train = data[train_begin:train_end]
    normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)
    train_x, train_y = [], []
    for i in range(len(normalized_train_data) - time_step):
        if i % batch_size == 0:
            batch_index.append(i)
        x = normalized_train_data[i: i + time_step, :7]
        y = normalized_train_data[i: i + time_step, 7, np.newaxis]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data) - time_step))
    return batch_index, train_x, train_y

# batch_size, train_x, train_y = get_train_data()
# print(batch_size)
# print('x', np.array(train_x, np.float32).shape)
# print('y', np.array(train_y, np.float32).shape)

# 获取测试集
def get_val_data(time_step=20, test_begin=5800):
    data_test = data[test_begin:]
    normalized_test_data = (data_test - np.mean(data_test, axis=0)) / np.std(data_test, axis=0)
    size = (len(normalized_test_data) + time_step - 1) / time_step
    test_x, test_y = [], []
    for i in range(size - 1):
        x = normalized_test_data[i * time_step: (i + 1) * time_step, : 7]
        y = normalized_test_data[i * time_step: (i + 1) * time_step, 7]
        test_x.append(x.tolist())
        test_y.append(y)
    test_x.append((normalized_test_data[(i + 1) * time_step:, :7]).tolist())
    test_y.extend((normalized_test_data[(i + 1) * time_step:, 7]).tolist())
    return test_x, test_y

weights = {
    'in': tf.Variable(tf.random_normal([INPUT_SIZE, HIDDEN_SIZE])),
    'out': tf.Variable(tf.random_normal([HIDDEN_SIZE, 1]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[HIDDEN_SIZE, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}

# 构建神经网络
def lstm(x):
    batch_size = tf.shape(x)[0]
    time_step = tf.shape(x)[1]
    w_in = weights['in']
    b_in = biases['in']

    inputs = tf.reshape(x, [-1, INPUT_SIZE])
    input_rnn = tf.matmul(inputs, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, HIDDEN_SIZE])

    cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
    init_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    output = tf.reshape(output_rnn, [-1, HIDDEN_SIZE])

    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out

    return pred, final_states

# 训练模型
def train_lstm(batch_size=80, time_step=15, train_begin=0, train_end=5800):
    X = tf.placeholder(tf.float32, shape=[None, time_step, INPUT_SIZE])
    Y = tf.placeholder(tf.float32, shape=[None, time_step, OUTPUT_SIZE])
    batch_index, train_x, train_y = get_train_data(batch_size, time_step, train_begin, train_end)

    pred, _ = lstm(X)
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            for step in range(len(batch_index) - 1):
                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[batch_index[step]: batch_index[step + 1]],
                                                                 Y: train_y[batch_index[step]: batch_index[step + 1]]})
            if not i % 100:
                print('%d steps, loss is: %f' % (i, loss_))

train_lstm()