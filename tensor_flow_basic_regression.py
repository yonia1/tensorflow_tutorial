import tensorflow as tf
import numpy as np


x_samples = np.random.rand(1000).astype(np.float32)
y_samples = x_samples * 2.0 + 1.5

weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
intercept = tf.Variable(tf.zeros([1]))
y = weights * x_samples + intercept

our_loss_function = tf.reduce_mean(tf.square(y - y_samples))
optimize = tf.train.GradientDescentOptimizer(0.5)
#optimize = tf.train.AdagradOptimizer(0.5)
#0.5 is the learning rate - how fast will we move to the goal
train = optimize.minimize(our_loss_function)

init = tf.global_variables_initializer()

session = tf.Session()
session.run(init)

for step in range(1000):
    session.run(train)

print(session.run(weights), session.run(intercept))
session.close()
