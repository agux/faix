from tensorflow.examples.tutorials.mnist import input_data
print("downloading mnist data...")
mnist = input_data.read_data_sets(
    "/tmp/tensorflow/mnist/input_data", source_url="http://yann.lecun.com/exdb/mnist/", one_hot=True)

import tensorflow as tf
from tensorflow.contrib import rnn

# define constants
# unrolled through 28 time steps
time_steps = 28
# hidden LSTM units
num_units = 512
# rows of 28 pixels
n_input = 28
# learning rate for adam
learning_rate = 0.001
# mnist is meant to be classified in 10 classes(0-9).
n_classes = 10
# size of batch
batch_size = 128

# weights and biases of appropriate shape to accomplish above task
out_weights = tf.Variable(tf.random.normal([num_units, n_classes]))
out_bias = tf.Variable(tf.random.normal([n_classes]))

# defining placeholders
# input image placeholder
x = tf.compat.v1.placeholder("float", [None, time_steps, n_input])
# input label placeholder
y = tf.compat.v1.placeholder("float", [None, n_classes])
keep_prob = tf.compat.v1.placeholder(dtype=tf.float32)

# processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
input = tf.unstack(x, time_steps, 1)

# defining the network
lstm_layer = rnn.BasicLSTMCell(num_units, forget_bias=1)
dw_cell = rnn.DropoutWrapper(lstm_layer, output_keep_prob=keep_prob)
outputs, _ = rnn.static_rnn(dw_cell, input, dtype="float32")

# converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
prediction = tf.matmul(outputs[-1], out_weights) + out_bias

# loss_function
loss = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(
    logits=prediction, labels=tf.stop_gradient(y)))
# optimization
opt = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# model evaluation
correct_prediction = tf.equal(tf.argmax(input=prediction, axis=1), tf.argmax(input=y, axis=1))
accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))

# initialize variables
init = tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as sess:
    sess.run(init)
    iter = 1
    while iter < 800:
        print("iter %d" % iter)
        batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)

        batch_x = batch_x.reshape((batch_size, time_steps, n_input))

        sess.run(opt, feed_dict={x: batch_x, y: batch_y, keep_prob:0.6})

        if iter % 10 == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob:1.0})
            los = sess.run(loss, feed_dict={x: batch_x, y: batch_y, keep_prob:1.0})
            print("For iter ", iter)
            print("Accuracy ", acc)
            print("Loss ", los)
            print("__________________")

        iter = iter + 1
    # calculating test accuracy
    test_data = mnist.test.images.reshape((-1, time_steps, n_input))
    test_label = mnist.test.labels
    print("Testing Accuracy:", sess.run(
        accuracy, feed_dict={x: test_data, y: test_label, keep_prob: 1.0}))
