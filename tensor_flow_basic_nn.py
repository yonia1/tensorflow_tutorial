import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import matplotlib.image as mpimg

session = tf.Session()

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
image = mnist.train.next_batch(1)

tmp = mnist.train.images[1]
tmp = tmp.reshape((28, 28))

plt.imshow(tmp)

x_input = tf.placeholder(tf.float32, [None, 784])

weights = tf.Variable(tf.zeros([784, 10]))
bias = tf.Variable(tf.zeros([10]))

y_predicted = tf.nn.softmax(tf.matmul(x_input, weights) + bias)

y_actual = tf.placeholder(tf.float32, [None, 10])

optimize_function = tf.reduce_mean(-tf.reduce_sum(y_actual * tf.log(y_predicted), reduction_indices=[1]))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for step in [0.01, 0.1, 0.5, 1, 2]:
    train_step = tf.train.GradientDescentOptimizer(step).minimize(optimize_function)
    for i in range(5000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x_input: batch_xs, y_actual: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y_predicted, 1), tf.argmax(y_actual, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x_input: mnist.test.images, y_actual: mnist.test.labels}))

# Creating a better NN for better results
sess.close()



def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


x_image = tf.reshape(x_input, [-1,28,28,1])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_actual))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_actual,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())



for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x_input:batch[0], y_actual: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x_input: batch[0], y_actual: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x_input: mnist.test.images, y_actual: mnist.test.labels, keep_prob: 1.0}))