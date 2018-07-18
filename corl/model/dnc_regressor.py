from __future__ import print_function
# Path hack.
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../..")

import tensorflow as tf
import numpy as np
import math
from pstk.model.model import lazy_property, stddev
from dnc import dnc

# pylint: disable-msg=E1101


class DNCRegressorV1:
    '''
    A Differentiable Neural Computer (DNC) Regressor.
    '''

    def __init__(self, layer_width=200, memory_size=16, word_size=16,
                 num_writes=1, num_reads=4, clip_value=20, max_grad_norm=50,
                 keep_prob=None, learning_rate=1e-3):
        self._layer_width = layer_width
        self._memory_size = memory_size
        self._word_size = word_size
        self._num_writes = num_writes
        self._num_reads = num_reads
        self._clip_value = clip_value
        self._max_grad_norm = max_grad_norm
        self._learning_rate = learning_rate
        self._keep_prob = keep_prob
        self._c_recln = 1
        self.recln_ops = {
            "selu": lambda layer: tf.nn.selu(layer),
            "dropout": lambda layer: tf.contrib.nn.alpha_dropout(
                layer, keep_prob=keep_prob),
        }

    def setNodes(self, features, target, seqlen, train_batch_size):
        self.data = features
        self.target = target
        self.seqlen = seqlen
        self._train_batch_size = train_batch_size
        self.logits
        self.optimize
        self.cost
        self.worst

    def getName(self):
        return self.__class__.__name__

    @lazy_property
    def logits(self):
        layer = self.dnc(self, self.data)
        layer = self.fcn(self, layer)
        with tf.variable_scope("output"):
            output = tf.layers.dense(
                inputs=layer,
                units=1,
                kernel_initializer=tf.variance_scaling_initializer(),
                bias_initializer=tf.constant_initializer(0.1)
            )
            output = tf.squeeze(output)
            # restoring shape info for the tensor
            output.set_shape([None])
            return output

    @staticmethod
    def fcn(self, inputs):
        layer = self.recln(self, inputs, ["selu"])
        fsize = int(inputs.get_shape()[-1])
        with tf.variable_scope("fcn"):
            nlayer = 3
            for i in range(nlayer):
                i = i+1
                layer = tf.layers.dense(
                    inputs=layer,
                    units=fsize,
                    kernel_initializer=tf.variance_scaling_initializer(),
                    bias_initializer=tf.constant_initializer(0.1)
                )
                if i == 1:
                    layer = self.recln(self, layer, ["dropout"])
                fsize = fsize // 2
        layer = self.recln(self, layer, ["selu"])
        return layer

    @staticmethod
    def recln(self, layer, ops=[]):
        with tf.variable_scope("rec_linear_{}".format(self._c_recln)):
            for op in ops:
                layer = self.recln_ops[op](layer)
            self._c_recln = self._c_recln + 1
            return layer

    @staticmethod
    def dnc(self, inputs):
        access_config = {
            "memory_size": self._memory_size,
            "word_size": self._word_size,
            "num_reads": self._num_reads,
            "num_writes": self._num_writes,
        }
        controller_config = {
            "hidden_size": self._layer_width,
        }
        dnc_core = dnc.DNC(access_config, controller_config,
                           self._layer_width, self._clip_value)
        initial_state = dnc_core.initial_state(inputs.get_shape()[0].value)
        # transpose to time major: [time, batch, feature]
        # tm_inputs = tf.transpose(inputs, perm=[1, 0, 2])
        output_sequence, _ = tf.nn.dynamic_rnn(
            cell=dnc_core,
            inputs=inputs,
            initial_state=initial_state,
            dtype=tf.float32,  # If there is no initial_state, you must give a dtype
            # time_major=True,
            sequence_length=self.seqlen)
        # layer = tf.concat(layer, 1)
        # restore to batch major: [batch, time, feature]
        # output_sequence = tf.transpose(output_sequence, perm=[1, 0, 2])
        output = self.last_relevant(output_sequence, self.seqlen)
        return output

    @staticmethod
    def last_relevant(output, length):
        with tf.name_scope("last_relevant"):
            batch_size = tf.shape(output)[0]
            relevant = tf.gather_nd(output, tf.stack(
                [tf.range(batch_size), length-1], axis=1))
            return relevant

    @lazy_property
    def cost(self):
        logits = self.logits
        print("checking shapes, target:{}, logits:{}".format(
            self.target.get_shape(), logits.get_shape()))
        with tf.name_scope("cost"):
            return tf.losses.mean_squared_error(labels=self.target, predictions=logits)

    @lazy_property
    def optimize(self):
        train_loss = self.cost
        with tf.name_scope("optimize"):
            # Set up optimizer with global norm clipping.
            trainable_variables = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(
                tf.gradients(train_loss, trainable_variables), self._max_grad_norm)
            optimizer = tf.train.AdamOptimizer(self._learning_rate)
            train_step = optimizer.apply_gradients(
                zip(grads, trainable_variables), global_step=tf.train.get_or_create_global_step())
            return train_step

    @lazy_property
    def worst(self):
        logits = self.logits
        with tf.name_scope("worst"):
            sqd = tf.squared_difference(logits, self.target)
            bidx = tf.argmax(sqd)
            max_diff = tf.sqrt(tf.reduce_max(sqd))
            predict = tf.gather(logits, bidx)
            actual = tf.gather(self.target, bidx)
            return max_diff, predict, actual
