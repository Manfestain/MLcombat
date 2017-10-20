import tensorflow as tf
import numpy as np

state = tf.Variable(0, name='counter')
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)
init_op = tf.global_variables_initializer()   # 启动图，对op进行初始化
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(state))
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))


with tf.Session() as sess:
    matrix1 = tf.constant([[2, 3]])
    matrix2 = tf.constant([[2], [3]])
    product = tf.matmul(matrix1, matrix2)
    result = sess.run(product)
    print(result)

input1 = tf.constant(3)
input2 = tf.constant(4)
input3 = tf.constant(5)
intermed = tf.add(input1, input2)
mul = tf.matmul(input3, intermed)
with tf.Session() as sess:
    result = sess.run([mul, intermed])
    print(result)