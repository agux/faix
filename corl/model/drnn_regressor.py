from __future__ import print_function
# Path hack.
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")

import tensorflow as tf
import numpy as np
import math
from pstk.model.model import lazy_property, stddev
from pstk.model.cells import AlphaDropoutWrapper
from rnn import grid_rnn

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


class DRnnRegressorV3:
    '''
    Deep RNN Regressor using 2 layers GridRNNCell and 3 layers FCN. Internal cell type is BasicLSTMCell.
    With batch norm, dropout, relu, and variance_scaling_initializer.
    '''

    def __init__(self, data=None, target=None, seqlen=None, layer_width=200, dim=3, training=None, learning_rate=1e-3):
        self.data = data
        self.target = target
        self.seqlen = seqlen
        self._layer_width = layer_width
        self._dim = dim
        self._learning_rate = learning_rate
        self._training = training
        if data is not None and target is not None and seqlen is not None:
            self.logits
            self.optimize
            self.cost
            self.worst

    def setNodes(self, uuids, features, target, seqlen):
        self.uuids = uuids
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
            return output

    @staticmethod
    def fcn(self, inputs):
        layer = inputs
        fsize = int(inputs.get_shape()[-1])
        with tf.variable_scope("fcn"):
            layer = tf.contrib.layers.batch_norm(
                inputs=layer,
                is_training=self._training,
                updates_collections=None
            )
            nlayer = 3
            for i in range(nlayer):
                i = i+1
                layer = tf.layers.dense(
                    inputs=layer,
                    units=fsize//i,
                    kernel_initializer=tf.variance_scaling_initializer(),
                    bias_initializer=tf.constant_initializer(0.1),
                    activation=tf.nn.relu
                )
            layer = tf.layers.dropout(
                inputs=layer,
                rate=0.5,
                training=self._training)
        return layer

    @staticmethod
    def newCell(width, _dim):
        def cell_fn(n):
            return tf.nn.rnn_cell.BasicLSTMCell(
                num_units=n
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
        layer = inputs
        nlayer = 2
        for i in range(nlayer):
            i = i+1
            with tf.variable_scope("rnn_{}".format(i)):
                layer, _ = tf.nn.dynamic_rnn(
                    self.newCell(self._layer_width * i, self._dim * i),
                    layer,
                    dtype=tf.float32,
                    sequence_length=self.seqlen if i < nlayer else None
                )
                layer = tf.concat(layer, 1)
                if i < nlayer:
                    layer = tf.contrib.layers.batch_norm(
                        inputs=layer,
                        is_training=self._training,
                        updates_collections=None
                    )
        output = self.last_relevant(layer, self.seqlen)
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
            self.cost, global_step=tf.train.get_or_create_global_step())

    @lazy_property
    def worst(self):
        logits = self.logits
        with tf.name_scope("worst"):
            sqd = tf.squared_difference(logits, self.target)
            bidx = tf.argmax(sqd)
            max_diff = tf.sqrt(tf.reduce_max(sqd))
            uuid = tf.gather(self.uuids, bidx)
            predict = tf.gather(logits, bidx)
            actual = tf.gather(self.target, bidx)
            return uuid, max_diff, predict, actual


class DRnnRegressorV4:
    '''
    Deep RNN Regressor using 1-layer GridRNNCell and 3-layer FCN. Internal cell type is BasicLSTMCell.
    With selu, and variance_scaling_initializer.
    '''

    def __init__(self, data=None, target=None, seqlen=None, layer_width=200, dim=3, keep_prob=None, learning_rate=1e-3):
        self.data = data
        self.target = target
        self.seqlen = seqlen
        self._layer_width = layer_width
        self._dim = dim
        self._learning_rate = learning_rate
        self._keep_prob = keep_prob
        self._c_recln = 1
        if data is not None and target is not None and seqlen is not None:
            self.logits
            self.optimize
            self.cost
            self.worst
        self.recln_ops = {
            "selu": lambda layer: tf.nn.selu(layer),
            "ln": lambda layer: tf.contrib.layers.layer_norm(layer, begin_norm_axis=0),
            "dropout": lambda layer: tf.contrib.nn.alpha_dropout(
                layer, keep_prob=keep_prob),
        }

    def setNodes(self, uuids, features, target, seqlen):
        self.uuids = uuids
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
                fsize = fsize // 2
        layer = self.recln(self, layer, ["selu"])
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
    def recln(self, layer, ops=[]):
        with tf.variable_scope("rec_linear_{}".format(self._c_recln)):
            for op in ops:
                layer = self.recln_ops[op](layer)
            self._c_recln = self._c_recln + 1
            return layer

    @staticmethod
    def rnn(self, inputs):
        layer = inputs
        layer, _ = tf.nn.dynamic_rnn(
            self.newCell(self._layer_width, self._dim),
            layer,
            dtype=tf.float32,
            sequence_length=self.seqlen
        )
        layer = tf.concat(layer, 1)
        output = self.last_relevant(layer, self.seqlen)
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
            self.cost, global_step=tf.train.get_or_create_global_step())

    @lazy_property
    def worst(self):
        logits = self.logits
        with tf.name_scope("worst"):
            sqd = tf.squared_difference(logits, self.target)
            bidx = tf.argmax(sqd)
            max_diff = tf.sqrt(tf.reduce_max(sqd))
            uuid = tf.gather(self.uuids, bidx)
            predict = tf.gather(logits, bidx)
            actual = tf.gather(self.target, bidx)
            return uuid, max_diff, predict, actual


class DRnnRegressorV5:
    '''
    Deep RNN Regressor using 1-layer GridRNNCell and 3-layer FCN. Internal cell type is LSTMBlockCell.
    With selu, layer norm, alpha_dropout, and variance_scaling_initializer.
    '''

    def __init__(self, data=None, target=None, seqlen=None, layer_width=200, dim=3, keep_prob=None, learning_rate=1e-3):
        self.data = data
        self.target = target
        self.seqlen = seqlen
        self._layer_width = layer_width
        self._dim = dim
        self._learning_rate = learning_rate
        self._keep_prob = keep_prob
        self._c_recln = 1
        if data is not None and target is not None and seqlen is not None:
            self.logits
            self.optimize
            self.cost
            self.worst
        self.recln_ops = {
            "selu": lambda layer: tf.nn.selu(layer),
            "ln": lambda layer: tf.contrib.layers.layer_norm(layer, begin_norm_axis=0),
            "dropout": lambda layer: tf.contrib.nn.alpha_dropout(
                layer, keep_prob=keep_prob),
        }

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
            return output

    @staticmethod
    def fcn(self, inputs):
        layer = self.recln(self, inputs, ["ln", "selu"])
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
    def recln(self, layer, ops=[]):
        with tf.variable_scope("rec_linear_{}".format(self._c_recln)):
            for op in ops:
                layer = self.recln_ops[op](layer)
            self._c_recln = self._c_recln + 1
            return layer

    @staticmethod
    def rnn(self, inputs):
        layer = inputs
        layer, _ = tf.nn.dynamic_rnn(
            self.newCell(self._layer_width, self._dim),
            layer,
            dtype=tf.float32,
            sequence_length=self.seqlen
        )
        layer = tf.concat(layer, 1)
        output = self.last_relevant(layer, self.seqlen)
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
            self.cost, global_step=tf.train.get_or_create_global_step())

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


class DRnnRegressorV6:
    '''
    Deep RNN Regressor using 1-layer GridRNNCell and 3-layer FCN. Internal cell type is LSTMBlockCell.
    With selu, alpha_dropout, and variance_scaling_initializer.
    '''

    def __init__(self, data=None, target=None, seqlen=None, layer_width=200, dim=3, keep_prob=None, learning_rate=1e-3):
        self.data = data
        self.target = target
        self.seqlen = seqlen
        self._layer_width = layer_width
        self._dim = dim
        self._learning_rate = learning_rate
        self._keep_prob = keep_prob
        self._c_recln = 1
        if data is not None and target is not None and seqlen is not None:
            self.logits
            self.optimize
            self.cost
            self.worst
        self.recln_ops = {
            "selu": lambda layer: tf.nn.selu(layer),
            "dropout": lambda layer: tf.contrib.nn.alpha_dropout(
                layer, keep_prob=keep_prob),
        }

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
    def recln(self, layer, ops=[]):
        with tf.variable_scope("rec_linear_{}".format(self._c_recln)):
            for op in ops:
                layer = self.recln_ops[op](layer)
            self._c_recln = self._c_recln + 1
            return layer

    @staticmethod
    def rnn(self, inputs):
        layer = inputs
        layer, _ = tf.nn.dynamic_rnn(
            self.newCell(self._layer_width, self._dim),
            layer,
            dtype=tf.float32,
            sequence_length=self.seqlen
        )
        layer = tf.concat(layer, 1)
        output = self.last_relevant(layer, self.seqlen)
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
        return tf.train.AdamOptimizer(self._learning_rate).minimize(
            self.cost, global_step=tf.train.get_or_create_global_step())

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


class DRnnRegressorV7:
    '''
    Deep RNN Regressor using 1-layer bespoke GridRNNCell and 3-layer FCN. Internal cell type is LSTMBlockCell.
    With selu, alpha_dropout, and variance_scaling_initializer, incorporated cosine_decay_restarts
    '''

    def __init__(self, layer_width=200, dim=3,
                 keep_prob=0.5, decayed_dropout_start=None, dropout_decay_steps=None,
                 learning_rate=1e-3, decayed_lr_start=None, lr_decay_steps=None, seed=None):
        self._layer_width = layer_width
        self._dim = dim
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
    def newCell(self):
        width = self._layer_width

        def cell_fn(n):
            return tf.contrib.rnn.LSTMBlockCell(
                num_units=width,
                use_peephole=True
            )
        grid = grid_rnn.GridRNNCell(
            num_units=width,
            num_dims=self._dim,
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
        return grid

    @staticmethod
    def rnn(self, inputs):
        layer = inputs
        layer, _ = tf.nn.dynamic_rnn(
            self.newCell(self),
            layer,
            dtype=tf.float32,
            sequence_length=self.seqlen
        )
        layer = tf.concat(layer, 1)
        output = self.last_relevant(layer, self.seqlen)
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
    def keep_prob(self):
        with tf.variable_scope("keep_prob"):
            gstep = tf.train.get_or_create_global_step()

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
        with tf.variable_scope("learning_rate"):
            gstep = tf.train.get_or_create_global_step()

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
        return tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.cost, global_step=tf.train.get_or_create_global_step())

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
