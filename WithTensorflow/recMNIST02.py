# _*_ coding:utf-8 _*_

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 定义神经网络的神经元数目
INPUT_NODE = 784
LAYER1_NODE = 500
OUTPUT_NODE = 10

# 每次训练数据的个数
BATCH_SIZE = 100

# 衰减学习率的参数
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99

# 正则化项的系数
REGULARIATION_RATE = 0.0001

# 滑动平均的参数
TRAINING_STEPS = 3000
MOVING_AVERAGE_DECAY = 0.99

# 定义神经网络和前向传播算法
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)

def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    # 定义神经网络的参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算前向传播结果
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 使用带有滑动平均的模型计算前行传播结果
    global_step = tf.Variable(0, trainable=False)
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_average.apply(tf.trainable_variables())

    average_y = inference(x, variable_average, weights1, biases1, weights2, biases2)

    # 计算交叉熵和损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularization

    # 使用衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.images.shape[0] / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )

    # 定义使用的优化方法
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 定义同时更新滑动平均值和参数的方法
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op('train')

    # 定义精度的计算
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话并开始训练
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        for i in range(TRAINING_STEPS):
            # 每1000次就在验证集上测试训练的模型精度
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print('After %d training step(s), validation accuracy using average model is %g' % (i, validate_acc))

            # 用于生成下一次迭代的训练数据
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        # 验证在测试集上的准确度
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print('After %d training step(S), test accuracy using average model is %g' % (TRAINING_STEPS, test_acc))

def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()