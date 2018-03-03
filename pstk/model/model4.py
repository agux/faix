from __future__ import print_function
import functools
import tensorflow as tf
import numpy as np
import math

from model import lazy_property
from cells import EGRUCell, EGRUCell_V1, EGRUCell_V2


class ERnnPredictorV1:

    def __init__(self, data, target, seqlen, height, training, dropout, num_hidden=200, num_layers=2, learning_rate=1e-4):
        self.data = data
        self.target = target
        self.seqlen = seqlen
        self._height = height
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
        egru = EGRUCell_V1(
            num_units=self._num_hidden,
            shape=[self._height, int(input.get_shape()[2])//self._height],
            kernel=[3, 3],
            kernel_initializer=tf.truncated_normal_initializer(
                stddev=0.01),
            bias_initializer=tf.constant_initializer(0.1)
        )

        cell = tf.nn.rnn_cell.MultiRNNCell([egru] * self._num_layers)

        output, self.training_state = tf.nn.dynamic_rnn(
            cell,
            input,
            dtype=tf.float32,
            sequence_length=self.seqlen,
            initial_state=self.training_state
        )

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


def getIndices(x, batch, n):
    print("x:{}".format(x.get_shape()))
    indices = tf.stack([tf.fill([batch], x[0]), [
        x[1]-n+i for i in range(n)]], axis=1)
    print(indices.get_shape())
    return indices


class ERnnPredictorV2:

    def __init__(self, data, target, seqlen, height, training, dropout, num_hidden=200, num_layers=2, learning_rate=1e-4):
        self.data = data
        self.target = target
        self.seqlen = seqlen
        self._height = height
        self.training = training
        self.dropout = dropout
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self._learning_rate = learning_rate
        self.prediction
        self.accuracy
        self.optimize

    @lazy_property
    def prediction(self):
        rnn = self.rnn(self, self.data)
        with tf.variable_scope("output"):
            dense = tf.contrib.bayesflow.layers.dense_flipout(
                inputs=rnn,
                units=rnn.get_shape()[1] * 2,
                activation=tf.nn.elu)
            # dropout = tf.layers.dropout(
            #     inputs=dense, rate=0.5, training=self.training)
            output = tf.contrib.bayesflow.layers.dense_flipout(
                inputs=dense,
                units=self.target.get_shape()[1])
                # kernel_initializer=tf.truncated_normal_initializer(
                #     stddev=0.01),
                # bias_initializer=tf.constant_initializer(0.1),
                # activation=tf.nn.elu)
            return output

    @staticmethod
    def rnn(self, input):
        # Recurrent network.
        egru = EGRUCell_V2(
            num_units=self._num_hidden,
            shape=[self._height, int(input.get_shape()[2])//self._height],
            kernel=[3, 3],
            kernel_initializer=tf.truncated_normal_initializer(
                stddev=0.01),
            bias_initializer=tf.constant_initializer(0.1)
        )
        cell = tf.nn.rnn_cell.MultiRNNCell([egru] * self._num_layers)
        output, _ = tf.nn.dynamic_rnn(
            cell,
            input,
            dtype=tf.float32,
            sequence_length=self.seqlen
        )
        output = self.last_n(output, 5, self.seqlen)
        # TODO gather last n output and convolute over them
        # output = conv(output, 2)
        output = tf.layers.flatten(output)
        return output

    @staticmethod
    def last_n(output, length, n):
        '''
        input:  [batch, max_step, feature]
        output: [batch, n, feature]
        where n is the last n features in max_step.
        '''
        with tf.variable_scope("last_n"):
            batch = tf.shape(output)[0]
            indices = tf.map_fn(lambda x: getIndices(x, batch, n),
                                tf.stack([tf.range(batch), length], axis=1))
            return tf.gather_nd(output, indices)

    @staticmethod
    def last_relevant(output, length):
        with tf.variable_scope("last_relevant"):
            batch_size = tf.shape(output)[0]
            relevant = tf.gather_nd(output, tf.stack(
                [tf.range(batch_size), length-1], axis=1))
            return relevant

    @lazy_property
    def cost(self):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.target, logits=self.prediction))
        kl = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss = cross_entropy + kl
        return loss

    @lazy_property
    def optimize(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            return tf.train.AdamOptimizer(self._learning_rate).minimize(
                self.cost, global_step=tf.train.get_global_step())

    @lazy_property
    def accuracy(self):
        accuracy = tf.equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(accuracy, tf.float32))
