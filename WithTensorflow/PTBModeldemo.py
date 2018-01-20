# _*_ coding:utf-8 _*_

import numpy as np
import tensorflow as tf
import pandas as pd
from  matplotlib import pyplot as plt

learn = tf.contrib.learn

HIDDEN_SIZE = 30
NUM_LAYERS = 2
TIMESTEPS = 10
TRAINING_STEPS = 10000
BATCH_SIZE = 32

TRAINING_EXAMPLES = 10000
TESTING_EXAMPLES = 1000
SAMPLE_GAP = 0.01

def generate_data(seq):
    X =[]
    y = []
    # print('seq:', seq[:20])
    for i in range(len(seq) - TIMESTEPS - 1):
        X.append([seq[i: i + TIMESTEPS]])
        y.append([seq[i + TIMESTEPS]])
    print('X: ', len(X))
    print('x: ', X[: 10])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def lstm_model(X, y):
    # 使用多层的LSTM结构
    inputs = tf.reshape(X, [-1, TIMESTEPS, HIDDEN_SIZE])
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * NUM_LAYERS)

    # 将多层LSTM结构连接成RNN网络并计算前向传播过程
    output, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    output = output[-1]

    prediction, loss = learn.models.linear_regression(output, y)

    # 创建优化器
    train_op = tf.contrib.layers.optimize_loss(
        loss,
        tf.contrib.framework.get_global_step(),
        optimizer='Adagrad',
        learning_rate=0.1
    )
    return prediction, loss, train_op

# 建立深层循环网络模型
regressor = learn.Estimator(model_fn=lstm_model)

test_start = TRAINING_EXAMPLES * SAMPLE_GAP
test_end = (TRAINING_EXAMPLES + TESTING_EXAMPLES) * SAMPLE_GAP
train_x, train_y = generate_data(np.sin(np.linspace(0, test_start, TESTING_EXAMPLES, dtype=np.float32)))
test_x, test_y = generate_data(np.sin(np.linspace(test_start, test_end, TESTING_EXAMPLES, dtype=np.float32)))

# 调用fit函数训练模型
input_x = tf.reshape(train_x, [-1, TIMESTEPS, HIDDEN_SIZE])
regressor.fit(input_x, train_y, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)

# 使用训练好的模型进行预测
predicted = [[pred] for pred in regressor.predict(test_x)]
rmse = np.sqrt(((predicted - test_y) ** 2).mean(axis=0))
print('Mean Square Error is: %f' % rmse[0])