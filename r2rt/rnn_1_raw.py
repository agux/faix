# source: https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html

import numpy as np
import tensorflow as tf
#matplotlib inline
import matplotlib.pyplot as plt

print("Expected cross entropy loss if the model:")
print("- learns neither dependency:", -(0.625 * np.log(0.625) +
                                        0.375 * np.log(0.375)))
# Learns first dependency only ==> 0.51916669970720941
print("- learns first dependency:  ",
      -0.5 * (0.875 * np.log(0.875) + 0.125 * np.log(0.125))
      - 0.5 * (0.625 * np.log(0.625) + 0.375 * np.log(0.375)))
print("- learns both dependencies: ", -0.50 * (0.75 * np.log(0.75) + 0.25 * np.log(0.25))
      - 0.25 * (2 * 0.50 * np.log(0.50)) - 0.25 * (0))


# Global config variables
# number of truncated backprop steps ('n' in the discussion above)
num_steps = 5
batch_size = 200
num_classes = 2
state_size = 4
learning_rate = 0.1


def gen_data(size=1000000):
    X = np.array(np.random.choice(2, size=(size,)))
    Y = []
    for i in range(size):
        threshold = 0.5
        if X[i - 3] == 1:
            threshold += 0.5
        if X[i - 8] == 1:
            threshold -= 0.25
        if np.random.rand() > threshold:
            Y.append(0)
        else:
            Y.append(1)
    return X, np.array(Y)

# adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py


def gen_batch(raw_data, batch_size, num_steps):
    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    # partition raw data into batches and stack them vertically in a data matrix
    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length *
                          i:batch_partition_length * (i + 1)]
        data_y[i] = raw_y[batch_partition_length *
                          i:batch_partition_length * (i + 1)]
    # further divide batch partitions into num_steps for truncated backprop
    epoch_size = batch_partition_length // num_steps

    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)


def gen_epochs(n, num_steps):
    for i in range(n):
        yield gen_batch(gen_data(), batch_size, num_steps)


"""
Placeholders
"""

x = tf.compat.v1.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
y = tf.compat.v1.placeholder(tf.int32, [batch_size, num_steps],
                   name='labels_placeholder')
init_state = tf.zeros([batch_size, state_size])

"""
RNN Inputs
"""

# Turn our x placeholder into a list of one-hot tensors:
# rnn_inputs is a list of num_steps tensors with shape [batch_size, num_classes]
x_one_hot = tf.one_hot(x, num_classes)
rnn_inputs = tf.unstack(x_one_hot, axis=1)

"""
Definition of rnn_cell

This is very similar to the __call__ method on Tensorflow's BasicRNNCell. See:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/python/ops/core_rnn_cell_impl.py#L95
"""
with tf.compat.v1.variable_scope('rnn_cell'):
    W = tf.compat.v1.get_variable('W', [num_classes + state_size, state_size])
    b = tf.compat.v1.get_variable('b', [state_size],
                        initializer=tf.compat.v1.constant_initializer(0.0))


def rnn_cell(rnn_input, state):
    with tf.compat.v1.variable_scope('rnn_cell', reuse=True):
        W = tf.compat.v1.get_variable('W', [num_classes + state_size, state_size])
        b = tf.compat.v1.get_variable('b', [state_size],
                            initializer=tf.compat.v1.constant_initializer(0.0))
    return tf.tanh(tf.matmul(tf.concat([rnn_input, state], 1), W) + b)


"""
Adding rnn_cells to graph

This is a simplified version of the "static_rnn" function from Tensorflow's api. See:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/python/ops/core_rnn.py#L41
Note: In practice, using "dynamic_rnn" is a better choice that the "static_rnn":
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn.py#L390
"""
state = init_state
rnn_outputs = []
for rnn_input in rnn_inputs:
    state = rnn_cell(rnn_input, state)
    rnn_outputs.append(state)
final_state = rnn_outputs[-1]

"""
Predictions, loss, training step

Losses is similar to the "sequence_loss"
function from Tensorflow's API, except that here we are using a list of 2D tensors, instead of a 3D tensor. See:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/seq2seq/python/ops/loss.py#L30
"""

#logits and predictions
with tf.compat.v1.variable_scope('softmax'):
    W = tf.compat.v1.get_variable('W', [state_size, num_classes])
    b = tf.compat.v1.get_variable('b', [num_classes],
                        initializer=tf.compat.v1.constant_initializer(0.0))
logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
predictions = [tf.nn.softmax(logit) for logit in logits]

# Turn our y placeholder into a list of labels
y_as_list = tf.unstack(y, num=num_steps, axis=1)

#losses and train_step
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logit) for
          logit, label in zip(logits, y_as_list)]
total_loss = tf.reduce_mean(input_tensor=losses) # sum of loss of each step
train_step = tf.compat.v1.train.AdagradOptimizer(learning_rate).minimize(total_loss)

"""
Train the network
"""


def train_network(num_epochs, num_steps, state_size=4, verbose=True):
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        training_losses = []
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps)):
            training_loss = 0
            training_state = np.zeros((batch_size, state_size))
            if verbose:
                print("\nEPOCH", idx)
            for step, (X, Y) in enumerate(epoch):
                tr_losses, training_loss_, training_state, _ = \
                    sess.run([losses,
                              total_loss,
                              final_state,
                              train_step],
                             feed_dict={x: X, y: Y, init_state: training_state})
                training_loss += training_loss_
                if step % 100 == 0 and step > 0:
                    if verbose:
                        print("Average loss at step", step,
                              "for last 250 steps:", training_loss / 100)
                    training_losses.append(training_loss / 100)
                    training_loss = 0

    return training_losses


training_losses = train_network(1, num_steps)
plt.plot(training_losses)
