from __future__ import print_function
# Path hack.
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")

import tensorflow as tf
import numpy as np
import math
from pstk.model.model import lazy_property, stddev

# pylint: disable-msg=E1101


class DRnnRegressorV1:
    '''
    Deep RNN Regressor using one of:
    GRU,
    GRUBlock,
    LSTM,
    BasicLSTM,
    LayerNormBasicLSTM,
    GLSTM,
    GridLSTM,
    LSTMBlock,
    UGRNN,
    NAS,
    etc...
    '''

    def __init__(self, data, target, seqlen, cell, use_peepholes=False, groups=1,
                 tied=False, layer_width=200, learning_rate=1e-3):
        self.data = data
        self.target = target
        self.seqlen = seqlen
        self._layer_width = layer_width
        self._learning_rate = learning_rate
        self._cell = cell
        self._use_peepholes = use_peepholes
        self._tied = tied
        self._groups = groups
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
            return output

    @staticmethod
    def fcn(self, inputs):
        layer = inputs
        with tf.variable_scope("fcn"):
            layer = tf.nn.relu(layer)
            for _ in range(5):
                width = int(layer.get_shape()[-1])//2
                layer = tf.layers.dense(
                    inputs=layer,
                    units=width,
                    kernel_initializer=tf.variance_scaling_initializer(),
                    bias_initializer=tf.constant_initializer(0.1)
                )
        return layer

    @staticmethod
    def newCell(self, _cell):
        c = None
        if _cell == 'gru':
            c = tf.nn.rnn_cell.GRUCell(
                num_units=self._layer_width
            )
        elif _cell == 'grublock':
            c = tf.contrib.rnn.GRUBlockCellV2(
                num_units=self._layer_width
            )
        elif _cell == 'lstm':
            c = tf.nn.rnn_cell.LSTMCell(
                num_units=self._layer_width,
                use_peepholes=self._use_peepholes
            )
        elif _cell == 'basiclstm':
            c = tf.nn.rnn_cell.BasicLSTMCell(
                num_units=self._layer_width
            )
        elif _cell == 'layernormbasiclstm':
            c = tf.contrib.rnn.LayerNormBasicLSTMCell(
                num_units=self._layer_width
            )
        elif _cell == 'glstm':
            c = tf.contrib.rnn.GLSTMCell(
                num_units=self._layer_width,
                number_of_groups=self._groups
            )
        elif _cell == 'gridlstm':
            c = tf.contrib.rnn.GridLSTMCell(
                num_units=self._layer_width,
                use_peephole=self._use_peepholes,
                share_time_frequency_weights=self._tied,
                num_unit_shards=self._groups
                # feature_size = feat_size,
                # num_frequency_blocks = ?
            )
        elif _cell == 'grid1lstm':
            c = tf.contrib.grid_rnn.Grid1LSTMCell(
                num_units=self._layer_width,
                use_peepholes=self._use_peepholes,
                output_is_tuple=False
            )
        elif _cell == 'grid2lstm':
            c = tf.contrib.grid_rnn.Grid2LSTMCell(
                num_units=self._layer_width,
                use_peepholes=self._use_peepholes,
                tied=self._tied,
                output_is_tuple=False
            )
        elif _cell == 'grid3lstm':
            c = tf.contrib.grid_rnn.Grid3LSTMCell(
                num_units=self._layer_width,
                use_peepholes=self._use_peepholes,
                tied=self._tied,
                output_is_tuple=False
            )
        elif _cell == 'grid2gru':
            c = tf.contrib.grid_rnn.Grid2GRUCell(
                num_units=self._layer_width,
                tied=self._tied,
                output_is_tuple=False
            )
        elif _cell == 'lstmblock':
            c = tf.contrib.rnn.LSTMBlockCell(
                num_units=self._layer_width,
                use_peephole=self._use_peepholes
            )
        elif _cell == 'nas':
            c = tf.contrib.rnn.NASCell(
                num_units=self._layer_width,
                use_biases=True
            )
        elif _cell == 'ugrnn':
            c = tf.contrib.rnn.UGRNNCell(
                num_units=self._layer_width
            )
        else:
            raise ValueError('unrecognized cell type:{}'.format(_cell))
        return c

    @staticmethod
    def rnn(self, inputs):
        cells = []
        _cell = self._cell.lower()
        for _ in range(2):
            cells.append(self.newCell(self, _cell))
        mc = tf.nn.rnn_cell.MultiRNNCell(cells)
        output, _ = tf.nn.dynamic_rnn(
            mc,
            inputs,
            dtype=tf.float32,
            sequence_length=self.seqlen
        )
        output = self.last_relevant(output, self.seqlen)
        print('last time step: {}'.format(output.get_shape()))
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
        with tf.name_scope("cost"):
            return tf.losses.mean_squared_error(labels=self.target, predictions=logits)

    @lazy_property
    def optimize(self):
        return tf.train.AdamOptimizer(self._learning_rate,
                                      epsilon=1e-7).minimize(
            self.cost, global_step=tf.train.get_global_step())

    @lazy_property
    def worst(self):
        logits = self.logits
        with tf.name_scope("worst"):
            sqd = tf.squared_difference(logits, self.target)
            bidx = tf.argmax(sqd)
            max_diff = tf.sqrt(tf.reduce_max(sqd))
            predict = tf.gather(logits, bidx)
            actual = tf.gather(self.target, bidx)
            return bidx, max_diff, predict, actual


class DRnnRegressorV2:
    '''
    Deep RNN Regressor using GridRNNCell, internal cell type is LSTMBlockCell.
    '''

    def __init__(self, data, target, seqlen, layer_width=200, dim=3, learning_rate=1e-3):
        self.data = data
        self.target = target
        self.seqlen = seqlen
        self._layer_width = layer_width
        self._dim = dim
        self._learning_rate = learning_rate
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
            return output

    @staticmethod
    def fcn(self, inputs):
        layer = inputs
        with tf.variable_scope("fcn"):
            layer = tf.nn.selu(layer)
            for _ in range(2):
                width = int(layer.get_shape()[-1])//2
                layer = tf.layers.dense(
                    inputs=layer,
                    units=width,
                    kernel_initializer=tf.variance_scaling_initializer(),
                    bias_initializer=tf.constant_initializer(0.1),
                    activation=tf.nn.selu
                )
        return layer

    @staticmethod
    def newCell(width, _dim):
        def cell_fn(n):
            return tf.contrib.rnn.LSTMBlockCell(
                num_units=n,
                use_peephole=True
            )
        c = tf.contrib.grid_rnn.GridRNNCell(
            num_units=width,
            num_dims=_dim,
            input_dims=0,
            output_dims=0,
            priority_dims=0,
            tied=False,
            non_recurrent_dims=None,
            cell_fn=cell_fn,
            non_recurrent_fn=None,
            state_is_tuple=True,
            output_is_tuple=True
        )
        return c

    @staticmethod
    def rnn(self, inputs):
        output, _ = tf.nn.dynamic_rnn(
            self.newCell(self._layer_width, self._dim),
            inputs,
            dtype=tf.float32,
            sequence_length=self.seqlen
        )
        output = tf.concat(output, 1)
        output = self.last_relevant(output, self.seqlen)
        print('last time step: {}'.format(output.get_shape()))
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
        with tf.name_scope("cost"):
            return tf.losses.mean_squared_error(labels=self.target, predictions=logits)

    @lazy_property
    def optimize(self):
        return tf.train.AdamOptimizer(self._learning_rate,
                                      epsilon=1e-7).minimize(
            self.cost, global_step=tf.train.get_global_step())

    @lazy_property
    def worst(self):
        logits = self.logits
        with tf.name_scope("worst"):
            sqd = tf.squared_difference(logits, self.target)
            bidx = tf.argmax(sqd)
            max_diff = tf.sqrt(tf.reduce_max(sqd))
            predict = tf.gather(logits, bidx)
            actual = tf.gather(self.target, bidx)
            return bidx, max_diff, predict, actual