from __future__ import print_function
import functools
import tensorflow as tf
import numpy as np
import math

from model import lazy_property, numLayers
from cells import EGRUCell, EGRUCell_V1, EGRUCell_V2


class MRnnPredictor:

    def __init__(self, data, target, seqlen, num_class, training, dropout, num_hidden=200, num_layers=1, learning_rate=1e-4):
        self.data = data
        self.target = target
        self.seqlen = seqlen
        self.training = training
        self.dropout = dropout
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self._num_class = num_class
        self._learning_rate = learning_rate
        self.prediction
        self.accuracy
        self.optimize
        self.cost

    @lazy_property
    def prediction(self):
        rnn = self.rnn(self, self.data)
        # ln = tf.contrib.layers.layer_norm(inputs=rnn)
        # dnn = self.dnn(self, rnn)
        output = tf.compat.v1.layers.dense(
            inputs=rnn,
            units=self._num_class,
            kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
            bias_initializer=tf.compat.v1.constant_initializer(0.1),
            activation=tf.nn.elu
        )
        return output

    @staticmethod
    def dnn(self, input):
        with tf.compat.v1.variable_scope("dnn"):
            dense = tf.compat.v1.layers.dense(
                inputs=input,
                units=self._num_hidden*2,
                kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                    stddev=0.01),
                bias_initializer=tf.compat.v1.constant_initializer(0.1)
                # activation=tf.nn.elu
            )
            dense = tf.compat.v1.layers.dense(
                inputs=dense,
                units=self._num_hidden,
                kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                    stddev=0.01),
                bias_initializer=tf.compat.v1.constant_initializer(0.1),
                # activation=tf.nn.elu
            )
            dense = tf.compat.v1.layers.dense(
                inputs=dense,
                units=self._num_hidden//2,
                kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                    stddev=0.01),
                bias_initializer=tf.compat.v1.constant_initializer(0.1),
                activation=tf.nn.relu6
            )
            dropout = tf.compat.v1.layers.dropout(
                inputs=dense, rate=0.8, training=self.training)
            return dropout

    @staticmethod
    def rnn(self, input):
        # Recurrent network.
        gru1 = tf.compat.v1.nn.rnn_cell.GRUCell(
            num_units=self._num_hidden,
            kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                stddev=0.01),
            bias_initializer=tf.compat.v1.constant_initializer(0.1)
        )
        gru2 = tf.compat.v1.nn.rnn_cell.GRUCell(
            num_units=self._num_hidden*2,
            kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                stddev=0.01),
            bias_initializer=tf.compat.v1.constant_initializer(0.1)
        )
        gru2 = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
            cell=gru2,
            output_keep_prob=(1.0-self.dropout)
        )
        cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([gru1, gru2] * self._num_layers)
        output, _ = tf.compat.v1.nn.dynamic_rnn(
            cell,
            input,
            dtype=tf.float32,
            sequence_length=self.seqlen
        )

        return self.last_relevant(output, self.seqlen)

    @staticmethod
    def last_relevant(output, length):
        batch_size = tf.shape(input=output)[0]
        relevant = tf.gather_nd(output, tf.stack(
            [tf.range(batch_size), length-1], axis=1))
        return relevant

    @lazy_property
    def cost(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.target, logits=self.prediction, name="xentropy")
        loss = tf.reduce_mean(input_tensor=cross_entropy)
        return loss

    @lazy_property
    def optimize(self):
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        optimizer = None
        with tf.control_dependencies(update_ops):
            optimizer = tf.compat.v1.train.AdamOptimizer(self._learning_rate).minimize(
                self.cost, global_step=tf.compat.v1.train.get_global_step())
        return optimizer

    @lazy_property
    def accuracy(self):
        accuracy = tf.equal(
            tf.argmax(input=self.target, axis=1), tf.argmax(input=self.prediction, axis=1))
        return tf.reduce_mean(input_tensor=tf.cast(accuracy, tf.float32), name="accuracy")
