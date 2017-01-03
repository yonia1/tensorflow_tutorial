import tensorflow as tf
import numpy as np

import imagesD
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Image details
image_width = 28
image_height = 28
total_pixels = image_width * image_height  # # img shape: 28*28
digit_classes = 10  # 0-9 digits


# Create placeholder for the input unknown size of images and 28*28 pixels
x = tf.placeholder(tf.float32, [None, total_pixels])
y = tf.placeholder(tf.float32, [None, digit_classes])


# keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)


def get_weights(shape):

    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def get_bias(length):
    return tf.Variable(tf.random_normal([length]))


# Create some wrappers for simplicity

def conv2d(x, w, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def conv_pull(x, w, b, strides=1):
    conv = conv2d(x, w, b)

    # Max Pooling (down-sampling)
    conv = maxpool2d(conv, k=2)
    return conv


# Create model
def magic_net(x):
    # Reshape input image to a 4 tensor
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # We would like a filter 5*5 , input is 1 image and output is 32 images that where filtered
    shape = [5, 5, 1, 32]
    # 1 Convolution Layer
    conv1 = conv_pull(x, get_weights(shape=shape), get_bias(length=32))
    # We would like a filter 5*5 , input is 32 images and output is 64 images that where filtered
    shape = [5, 5, 32, 64]
    #
    # 2 Convolution Layer
    conv2 = conv_pull(conv1, get_weights(shape=shape), get_bias(length=64))



    # Fully connected layer we move from 64 images size 7*7 *64 = 3136
    # Reshape conv2 output to fit fully connected layer input

    fully_connected_length_start = 7*7*64

    fully_connected_length_end = 1024

    w = get_weights([fully_connected_length_start, fully_connected_length_end])
    b = get_bias(fully_connected_length_end)
    fc1 = tf.reshape(conv2, [-1, w.get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, w), b)
    # convert negative to zero
    fc1 = tf.nn.relu(fc1)

    # Output, class prediction last layer is the prediction of the 10 classes
    out = tf.add(tf.matmul(fc1, get_weights([fully_connected_length_end, digit_classes])), get_bias(digit_classes))
    return out

mnist.test.cls= mnist.test.cls = np.argmax(mnist.test.labels, axis=1)
images = mnist.test.images[0:9]
print images
# Get the true classes for those images.
cls_true = mnist.test.cls[0:9]

# Plot the images and labels using our helper-function above.
imagesD.plot_images(images=images, cls_true=cls_true,img_shape=28)


# Construct model
pred = magic_net(x)

learning_rate = 0.001
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

iters = 2000000
batch_size = 150
display_step = 10

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)

        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1


    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
          sess.run(accuracy, feed_dict={x: mnist.test.images[:300],
                                        y: mnist.test.labels[:300]}))

    num_test = 300

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)
