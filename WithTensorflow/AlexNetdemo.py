# _*- coding:utf-8 _*_

import math
import time
import tensorflow as tf
from datetime import datetime

batch_size = 32
num_batches = 100

# 查看每一层网络结构
def print_activations(t):
    print(t.op.name, '  ', t.get_shape().as_list())

def inference(images):
    parameters = []

    # 第一个卷积层
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32, stddev=1e-1),
                             name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv1)

    lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn1')
    pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
    print_activations(pool1)

    # 第二个卷积层
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32, stddev=1e-1),
                             name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv2)

    lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn2')
    pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
    print_activations(pool2)

    # 第三个卷积层
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.random_normal([3, 3, 192, 384], dtype=tf.float32, stddev=1e-1),
                             name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv3)

    # 第四个卷积层
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32, stddev=1e-1),
                             name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv4)

    # 第五个卷积层
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1),
                             name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv5)

    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
    print_activations(pool5)

    # 第一个全连接层
    with tf.name_scope('fcl1') as scope:
        weight = tf.Variable(tf.truncated_normal([6 * 6 * 256, 4096], stddev=0.1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')
        h_pool5_flat = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fcl1 = tf.nn.relu(tf.matmul(h_pool5_flat, weight) + biases, name=scope)
        drop1 = tf.nn.dropout(fcl1, 0.7)
        parameters += [weight, biases]
        print_activations(fcl1)

    # 第二个全连接层
    with tf.name_scope('fcl2') as scope:
        weight = tf.Variable(tf.truncated_normal([4096, 4096], stddev=0.1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')
        fcl2 = tf.nn.relu(tf.matmul(drop1, weight) + biases, name=scope)
        drop2 = tf.nn.dropout(fcl2, 0.7)
        parameters += [weight, biases]
        print_activations(fcl2)

    # 第三个全连接层
    with tf.name_scope('fcl3') as scope:
        weight = tf.Variable(tf.truncated_normal([4096, 1000], stddev=0.1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[1000], dtype=tf.float32), trainable=True, name='biases')
        fcl3 = tf.nn.relu(tf.matmul(drop2, weight) + biases, name=scope)
        parameters += [weight, biases]
        print_activations(fcl3)

    return fcl3, parameters

# 时间评估
def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_durtaion = 0.0
    total_durtaion_squared = 0.0

    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d,  duration = %.3f' % (datetime.now(), i - num_steps_burn_in, duration))
            total_durtaion += duration
            total_durtaion_squared += duration * duration

    mn = total_durtaion / num_batches
    vr = total_durtaion_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' % (datetime.now(), info_string, num_batches, mn, sd))

# 主程序
def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size, image_size, image_size, 3], dtype=tf.float32, stddev=1e-1))
        pool5, parameters = inference(images)
        print(parameters)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        time_tensorflow_run(sess, pool5, 'Forward')

        objective = tf.nn.l2_loss(pool5)
        grad = tf.gradients(objective, parameters)
        time_tensorflow_run(sess, grad, "Forward_backward")

if __name__ == '__main__':
    run_benchmark()

