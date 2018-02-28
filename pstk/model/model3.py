from __future__ import print_function
import functools
import tensorflow as tf
import numpy as np
import math

from model import lazy_property


def primes(n):
    primfac = []
    d = 2
    while d*d <= n:
        while (n % d) == 0:
            primfac.append(d)  # supposing you want multiple factors repeated
            n //= d
        d += 1
    if n > 1:
        primfac.append(n)
    return primfac


def numLayers(d1, d2=None):
    n1 = 0
    while d1 > 1:
        d1 = math.ceil(d1/2)
        n1 += 1
    n2 = 0
    if d2 is not None:
        n2 = numLayers(d2)
    return max(n1, n2)


def factorize(feature_size):
    factors = sorted(primes(feature_size))
    if len(factors) < 2:
        raise ValueError('feature size is not factorizable.')
    return tuple(sorted([reduce(lambda x, y: x*y, factors[:-1]), factors[-1]], reverse=True))


def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = int(output.get_shape()[1])
    output_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, output_size])
    relevant = tf.gather(flat, index)
    return relevant


class RnnPredictorV1:

    def __init__(self, data, target, seqlen, training, dropout, num_hidden=200, num_layers=2, learning_rate=1e-4):
        self.data = data
        self.target = target
        self.seqlen = seqlen
        self.training = training
        self.dropout = dropout
        self.training_state = None
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self._learning_rate = learning_rate
        self.prediction
        self.accuracy
        self.optimize

    @lazy_property
    def prediction(self):
        rnn = self.rnn(self, self.data)
        dense = tf.layers.dense(
            inputs=rnn,
            units=self._num_hidden * 3,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            bias_initializer=tf.constant_initializer(0.1),
            activation=tf.nn.relu6)
        dropout = tf.layers.dropout(
            inputs=dense, rate=0.5, training=self.training)
        output = tf.layers.dense(
            inputs=dropout,
            units=int(self.target.get_shape()[1]),
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            bias_initializer=tf.constant_initializer(0.1),
            activation=tf.nn.elu)
        return output

    @staticmethod
    def rnn(self, input):
        # Recurrent network.
        cells = []
        for _ in range(self._num_layers):
            cell = tf.nn.rnn_cell.GRUCell(
                self._num_hidden,
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=0.01),
                bias_initializer=tf.constant_initializer(0.1))
            cells.append(cell)
            cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self._num_hidden,
                forget_bias=1.0,
                # input_size=None,
                activation=tf.tanh,
                layer_norm=True,
                # norm_gain=1.0,
                # norm_shift=1e-3,
                # dropout_keep_prob=1.0 - self.dropout,
                # dropout_prob_seed=None,
                # reuse=None
            )
            cells.append(cell)
            # cell = tf.nn.rnn_cell.DropoutWrapper(
            #     cell, output_keep_prob=1.0 - self.dropout)

        cell = tf.nn.rnn_cell.MultiRNNCell(cells)

        output, self.training_state = tf.nn.dynamic_rnn(
            cell,
            input,
            dtype=tf.float32,
            sequence_length=self.seqlen,
            initial_state=self.training_state
        )

        last = last_relevant(output, self.seqlen)
        return last

    @lazy_property
    def cost(self):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.target, logits=self.prediction))
        # cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
        return cross_entropy

    @lazy_property
    def optimize(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        optimizer = None
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(
                self.cost, global_step=tf.train.get_global_step())
        return optimizer

    @lazy_property
    def accuracy(self):
        accuracy = tf.equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(accuracy, tf.float32))


class RnnPredictorV2:

    def __init__(self, data, target, seqlen, width, training, dropout, num_hidden=200, num_layers=2, learning_rate=1e-4):
        self.data = data
        self.target = target
        self.seqlen = seqlen
        self._input_width = width
        self.training = training
        self.dropout = dropout
        self.training_state = None
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self._learning_rate = learning_rate
        self.prediction
        self.accuracy
        self.optimize

    @lazy_property
    def prediction(self):
        rnn = self.rnn(self, self.data)
        dense = tf.layers.dense(
            inputs=rnn,
            units=self._num_hidden * 3,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            bias_initializer=tf.constant_initializer(0.1),
            activation=tf.nn.relu6)
        dropout = tf.layers.dropout(
            inputs=dense, rate=0.5, training=self.training)
        output = tf.layers.dense(
            inputs=dropout,
            units=int(self.target.get_shape()[1]),
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            bias_initializer=tf.constant_initializer(0.1),
            activation=tf.nn.elu)
        return output

    @staticmethod
    def rnn(self, input):
        # Recurrent network.
        # TODO add tf.contrib.rnn.ConvLSTMCell?
        step = int(input.get_shape()[1])
        feat = int(input.get_shape()[2])
        c = feat // self._input_width  # channel
        # TODO step & width must equal?
        input = tf.reshape(input, [-1, step, self._input_width, c])
        clc = tf.contrib.rnn.ConvLSTMCell(
            conv_ndims=1,
            input_shape=[step, self._input_width],
            output_channels=self._num_hidden,
            kernel_shape=[3],
            use_bias=True,
            skip_connection=False,
            forget_bias=1.0,
            initializers=None
        )

        cell = tf.nn.rnn_cell.MultiRNNCell([clc] * self._num_layers)

        output, self.training_state = tf.nn.dynamic_rnn(
            cell,
            input,
            dtype=tf.float32,
            sequence_length=self.seqlen,
            initial_state=self.training_state
        )

        output = tf.reshape(output, [-1, step, self._input_width*self._num_hidden])

        return self.last_relevant(output, self.seqlen)

    @staticmethod
    def last_relevant(output, length):
        batch_size = tf.shape(output)[0]
        relevant = tf.gather_nd(output, tf.stack(
            [tf.range(batch_size), length-1], axis=1))
        return relevant

    @lazy_property
    def cost(self):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.target, logits=self.prediction))
        # cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
        return cross_entropy

    @lazy_property
    def optimize(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        optimizer = None
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(
                self.cost, global_step=tf.train.get_global_step())
        return optimizer

    @lazy_property
    def accuracy(self):
        accuracy = tf.equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(accuracy, tf.float32))
