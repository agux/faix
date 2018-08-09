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
from misc.msgr.memory_saving_gradients import gradients

# pylint: disable-msg=E1101


class DNCRegressorV1:
    '''
    A Differentiable Neural Computer (DNC) Regressor.
    '''

    def __init__(self, layer_width=200, memory_size=16, word_size=16,
                 num_writes=1, num_reads=4, clip_value=20, max_grad_norm=50,
                 keep_prob=0.5, decayed_dropout_start=None, dropout_decay_steps=None,
                 learning_rate=1e-3, decayed_lr_start=None, lr_decay_steps=None, seed=None):
        self._layer_width = layer_width
        self._memory_size = memory_size
        self._word_size = word_size
        self._num_writes = num_writes
        self._num_reads = num_reads
        self._clip_value = clip_value
        self._max_grad_norm = max_grad_norm
        self._kp = keep_prob
        self._decayed_dropout_start = decayed_dropout_start
        self._dropout_decay_steps = dropout_decay_steps
        self._lr = learning_rate
        self._decayed_lr_start = decayed_lr_start
        self._lr_decay_steps = lr_decay_steps
        self._seed = seed

        self.keep_prob
        self.learning_rate

    def setNodes(self, features, target, seqlen):
        self.data = features
        self.target = target
        self.seqlen = seqlen
        self.logits
        self.optimize
        self.cost
        self.worst

    def getName(self):
        return self.__class__.__name__

    @lazy_property
    def logits(self):
        layer = self.rnn(self, self.data)
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
        layer = tf.nn.selu(inputs)
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
                    layer = tf.contrib.nn.alpha_dropout(
                        layer, keep_prob=self.keep_prob)
                fsize = fsize // 2
        layer = tf.nn.selu(layer)
        return layer

    @staticmethod
    def rnn(self, inputs):
        access_config = {
            "memory_size": self._memory_size,
            "word_size": self._word_size,
            "num_reads": self._num_reads,
            "num_writes": self._num_writes,
        }
        controller_config = {
            # "num_layers": self._num_layers,
            "hidden_size": self._layer_width,
            "initializers": {
                "w_gates": tf.variance_scaling_initializer(),
                "b_gates": tf.constant_initializer(0.1),
                "w_f_diag": tf.variance_scaling_initializer(),
                "w_i_diag": tf.variance_scaling_initializer(),
                "w_o_diag": tf.variance_scaling_initializer()
            },
            "use_peepholes": True
        }
        with tf.variable_scope("RNN"):
            dnc_core = dnc.DNC(access_config, controller_config,
                               self._layer_width, self._clip_value)
            initial_state = dnc_core.initial_state(tf.shape(inputs)[0])
            # transpose to time major: [time, batch, feature]
            # tm_inputs = tf.transpose(inputs, perm=[1, 0, 2])
            output_sequence, _ = tf.nn.dynamic_rnn(
                cell=dnc_core,
                inputs=inputs,
                initial_state=initial_state,
                # parallel_iterations=256,
                # dtype=tf.float32,  # If there is no initial_state, you must give a dtype
                # time_major=True,
                # swap_memory=True,
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
    def keep_prob(self):
        gstep = tf.train.get_or_create_global_step()
        with tf.variable_scope("keep_prob"):
            def kp():
                return tf.multiply(self._kp, 1.0)

            def cdr_kp():
                return 1.0-tf.train.cosine_decay_restarts(
                    learning_rate=1.0-self._kp,
                    global_step=gstep-self._decayed_dropout_start,
                    first_decay_steps=self._dropout_decay_steps,
                    t_mul=1.05,
                    m_mul=0.98,
                    alpha=0.01
                )
            minv = kp()
            if self._decayed_dropout_start is not None:
                minv = tf.cond(
                    tf.less(gstep, self._decayed_dropout_start), kp, cdr_kp)
            rdu = tf.random_uniform(
                [],
                minval=minv,
                maxval=1.02,
                dtype=tf.float32,
                seed=self._seed
            )
            return tf.minimum(1.0, rdu)

    @lazy_property
    def learning_rate(self):
        gstep = tf.train.get_or_create_global_step()
        with tf.variable_scope("learning_rate"):
            def tslr():
                return tf.multiply(self._lr, 1.0)

            def cdr():
                return tf.train.cosine_decay_restarts(
                    learning_rate=self._lr,
                    global_step=gstep-self._decayed_lr_start,
                    first_decay_steps=self._lr_decay_steps,
                    t_mul=1.02,
                    m_mul=0.95,
                    alpha=0.095
                )
            if self._decayed_lr_start is None:
                return tslr()
            else:
                return tf.cond(tf.less(gstep, self._decayed_lr_start), tslr, cdr)

    @lazy_property
    def optimize(self):
        train_loss = self.cost
        with tf.name_scope("optimize"):
            # Set up optimizer with global norm clipping.
            trainable_variables = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(
                tf.gradients(train_loss, trainable_variables), self._max_grad_norm)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
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


class DNCRegressorV2:
    '''
    A Differentiable Neural Computer (DNC) Regressor with optional gradient-checkpointing.
    gdck = 'collection' / 'memory' / 'speed'.
    '''

    def __init__(self, layer_width=200, memory_size=16, word_size=16,
                 num_writes=1, num_reads=4, clip_value=20, max_grad_norm=50,
                 keep_prob=0.5, decayed_dropout_start=None, dropout_decay_steps=None,
                 learning_rate=1e-3, decayed_lr_start=None, lr_decay_steps=None, seed=None,
                 gdck='memory'):
        self._layer_width = layer_width
        self._memory_size = memory_size
        self._word_size = word_size
        self._num_writes = num_writes
        self._num_reads = num_reads
        self._clip_value = clip_value
        self._max_grad_norm = max_grad_norm
        self._kp = keep_prob
        self._decayed_dropout_start = decayed_dropout_start
        self._dropout_decay_steps = dropout_decay_steps
        self._lr = learning_rate
        self._decayed_lr_start = decayed_lr_start
        self._lr_decay_steps = lr_decay_steps
        self._seed = seed
        self._gdck=gdck

        self.keep_prob
        self.learning_rate

    def setNodes(self, features, target, seqlen):
        self.data = features
        self.target = target
        self.seqlen = seqlen
        self.logits
        self.optimize
        self.cost
        self.worst

    def getName(self):
        return self.__class__.__name__

    @lazy_property
    def logits(self):
        layer = self.rnn(self, self.data)
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
        layer = tf.nn.selu(inputs)
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
                    layer = tf.contrib.nn.alpha_dropout(
                        layer, keep_prob=self.keep_prob)
                fsize = fsize // 2
        layer = tf.nn.selu(layer)
        return layer

    @staticmethod
    def rnn(self, inputs):
        access_config = {
            "memory_size": self._memory_size,
            "word_size": self._word_size,
            "num_reads": self._num_reads,
            "num_writes": self._num_writes,
        }
        controller_config = {
            # "num_layers": self._num_layers,
            "hidden_size": self._layer_width,
            "initializers": {
                "w_gates": tf.variance_scaling_initializer(),
                "b_gates": tf.constant_initializer(0.1),
                "w_f_diag": tf.variance_scaling_initializer(),
                "w_i_diag": tf.variance_scaling_initializer(),
                "w_o_diag": tf.variance_scaling_initializer()
            },
            "use_peepholes": True
        }
        with tf.variable_scope("RNN"):
            dnc_core = dnc.DNC(access_config, controller_config,
                               self._layer_width, self._clip_value)
            initial_state = dnc_core.initial_state(tf.shape(inputs)[0])
            output_sequence, _ = tf.nn.dynamic_rnn(
                cell=dnc_core,
                inputs=inputs,
                initial_state=initial_state,
                sequence_length=self.seqlen)
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
    def keep_prob(self):
        gstep = tf.train.get_or_create_global_step()
        with tf.variable_scope("keep_prob"):
            def kp():
                return tf.multiply(self._kp, 1.0)

            def cdr_kp():
                return 1.0-tf.train.cosine_decay_restarts(
                    learning_rate=1.0-self._kp,
                    global_step=gstep-self._decayed_dropout_start,
                    first_decay_steps=self._dropout_decay_steps,
                    t_mul=1.005,
                    m_mul=0.998,
                    alpha=0.05
                )
            minv = kp()
            if self._decayed_dropout_start is not None:
                minv = tf.cond(
                    tf.less(gstep, self._decayed_dropout_start), kp, cdr_kp)
            rdu = tf.random_uniform(
                [],
                minval=minv,
                maxval=1.02,
                dtype=tf.float32,
                seed=self._seed
            )
            return tf.minimum(1.0, rdu)

    @lazy_property
    def learning_rate(self):
        gstep = tf.train.get_or_create_global_step()
        with tf.variable_scope("learning_rate"):
            def tslr():
                return tf.multiply(self._lr, 1.0)

            def cdr():
                return tf.train.cosine_decay_restarts(
                    learning_rate=self._lr,
                    global_step=gstep-self._decayed_lr_start,
                    first_decay_steps=self._lr_decay_steps,
                    t_mul=1.002,
                    m_mul=0.995,
                    alpha=0.09
                )
            if self._decayed_lr_start is None:
                return tslr()
            else:
                return tf.cond(tf.less(gstep, self._decayed_lr_start), tslr, cdr)

    @lazy_property
    def optimize(self):
        train_loss = self.cost
        with tf.name_scope("optimize"):
            # Set up optimizer with global norm clipping.
            trainable_variables = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(
                gradients(train_loss, trainable_variables, checkpoints=self._gdck),
                # tf.gradients(train_loss, trainable_variables), 
                self._max_grad_norm)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
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
