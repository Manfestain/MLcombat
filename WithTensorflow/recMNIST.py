# _*_ coding:utf-8 _*_

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def loadData():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    return mnist

def main():
    mnist = loadData()

    sess = tf.InteractiveSession()   # 注册为默认的session

    x = tf.placeholder(tf.float32, [None, 784])   # 定义一个输入数据的地方
    W = tf.Variable(tf.zeros([784, 10]))   # 定义一个变量，用来迭代和训练权值
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)   # 定义一个激活函数

    y_ = tf.placeholder(tf.float32, [None, 10])   #
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
                                                  reduction_indices=[1]))   # 计算交叉损失
    #   使用随机梯度下降优化算法
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    tf.global_variables_initializer().run()

    #   进行1000次迭代
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)   # 使用一小部分数据进行随机梯度下降
        train_step.run({x: batch_xs, y_: batch_ys})
        
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == '__main__':
    main()