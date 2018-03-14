from __future__ import print_function
import functools
import tensorflow as tf
import numpy as np
import math

from model import lazy_property, numLayers
from cells import EGRUCell, EGRUCell_V1, EGRUCell_V2


class MRnnPredictor:

    def __init__(self, data, target, seqlen, training, dropout, num_hidden=200, num_layers=1, learning_rate=1e-4):
        self.data = data
        self.target = target
        self.seqlen = seqlen
        self.training = training
        self.dropout = dropout
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self._num_class = int(target.get_shape()[1])
        self._learning_rate = learning_rate
        self.prediction
        self.accuracy
        self.optimize
        self.cost

    @lazy_property
    def prediction(self):
        rnn = self.rnn(self, self.data)
        ln = tf.contrib.layers.layer_norm(inputs=rnn)
        dnn = self.dnn(self, ln)
        output = tf.layers.dense(
            inputs=dnn,
            units=self._num_class,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            bias_initializer=tf.constant_initializer(0.1),
            activation=tf.nn.elu)
        return output

    @staticmethod
    def dnn(self, input):
        with tf.variable_scope("dnn"):
            dense = tf.layers.dense(
                inputs=input,
                units=self._num_hidden*2,
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=0.01),
                bias_initializer=tf.constant_initializer(0.1),
                activation=tf.nn.elu
            )
            dense = tf.layers.dense(
                inputs=dense,
                units=self._num_hidden*4,
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=0.01),
                bias_initializer=tf.constant_initializer(0.1)
            )
            dropout = tf.layers.dropout(
                inputs=dense, rate=0.5, training=self.training)
            dense = tf.layers.dense(
                inputs=dropout,
                units=self._num_class * 4,
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=0.01),
                bias_initializer=tf.constant_initializer(0.1),
                activation=tf.nn.relu6)
            return dense

    @staticmethod
    def rnn(self, input):
        # Recurrent network.
        gru1 = tf.nn.rnn_cell.GRUCell(
            num_units=self._num_hidden * 2,
            kernel_initializer=tf.truncated_normal_initializer(
                stddev=0.01),
            bias_initializer=tf.constant_initializer(0.1)
        )
        gru1 = tf.nn.rnn_cell.DropoutWrapper(
            cell=gru1,
            variational_recurrent=True,
            output_keep_prob=self.dropout
        )
        gru2 = tf.nn.rnn_cell.GRUCell(
            num_units=self._num_hidden,
            kernel_initializer=tf.truncated_normal_initializer(
                stddev=0.01),
            bias_initializer=tf.constant_initializer(0.1)
        )
        cell = tf.nn.rnn_cell.MultiRNNCell([gru1, gru2] * self._num_layers)
        output, _ = tf.nn.dynamic_rnn(
            cell,
            input,
            dtype=tf.float32,
            sequence_length=self.seqlen
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
