# _*_ coding:utf-8 _*_

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import metrics

num_steps = 100

# 简单循环神经网络的计算过程
def simpleRNN():

    X = [1, 2]
    state = [0.0, 0.0]

    w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])
    w_cell_input = np.asarray([0.5, 0.6])
    b_cell = np.asarray([0.1, -0.1])

    w_output = np.asarray([1.0, 2.0])
    b_output = 0.1

    for i in range(len(X)):
        before_activation = np.dot(state, w_cell_state) + X[i] * w_cell_input + b_cell
        state = np.tanh(before_activation)
        final_output = np.dot(state, w_output) + b_output
        print('before activation: ', before_activation)
        print('state: ', state)
        print('output: ', final_output)
        print('\n')

# # 使用Tensorflow实现LSTM
# def simpleLSTM():
#     lstm = tf.contrib.rnn.BasicLSTMCell()
#     state = lstm.zero_state(batch_size=32, dtype=tf.float32)
#     loss= 0.0
#     for i in range(num_steps):
#         if i > 0:
#             tf.get_variable_scope().reuse_variables()
#             lstm_output, state = lstm(current_input, state)
#             final_output = fully_connected(lstm_output)
#
#             loss += calc_loss(final_output, expected_output)


learn = tf.contrib.learn
def my_model(features, target):
    target = tf.one_hot(target, 3, 1, 0)
    logits, loss = learn.models.logistic_regression(features, target)
    train_op = tf.contrib.layers.optimize_loss(
        loss,
        tf.contrib.framework.get_global_step(),
        optimizer='Adagrad',
        learning_rate=0.1
    )
    return tf.arg_max(logits, 1), loss, train_op

iris = datasets.load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)
classifier = learn.Estimator(model_fn=my_model)
classifier.fit(x_train, y_train, steps=100)
y_predicted = classifier.predict(x_test, as_iterable=False)
print(y_test)
score = metrics.accuracy_score(y_test, y_predicted)
print('Accuracy: %.3f%%' % (score * 100))



if __name__ == '__main__':
    simpleRNN()