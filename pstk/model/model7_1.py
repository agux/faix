from __future__ import print_function
import functools
import tensorflow as tf
import numpy as np
import math
from metrics import precision, recall
from model import lazy_property, residual, stddev
from cells import LayerNormNASCell, LayerNormGRUCell

# pylint: disable-msg=E1101


class SRnnPredictorV2:
    '''
    Simple RNN Predictor using one of:
    GRU,
    GRUBlock,
    LSTM,
    BasicLSTM,
    LayerNormBasicLSTM,
    GLSTM,
    GridLSTM,
    LSTMBlock,
    UGRNN,
    NAS
    '''

    def __init__(self, data, target, seqlen, classes, cell, use_peepholes=False, groups=1,
                 tied=False, layer_width=200, learning_rate=1e-3):
        self.data = data
        self.target = target
        self.seqlen = seqlen
        self._layer_width = layer_width
        self._classes = classes
        self._learning_rate = learning_rate
        self._cell = cell
        self._use_peepholes = use_peepholes
        self._tied = tied
        self._groups = groups
        self.precisions
        self.recalls
        self.f_score
        self.logits
        self.accuracy
        self.optimize
        self.cost
        self.one_hot
        self.worst

    def getName(self):
        return self.__class__.__name__

    @lazy_property
    def logits(self):
        layer = self.rnn(self, self.data)
        output = tf.compat.v1.layers.dense(
            inputs=layer,
            units=len(self._classes),
            kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                stddev=stddev(1.0, int(layer.get_shape()[-1]))),
            bias_initializer=tf.compat.v1.constant_initializer(0.1),
            activation=tf.nn.relu6,
            name="output"
        )
        return output

    @staticmethod
    def rnn(self, inputs):
        c = None
        _cell = self._cell.lower()
        if _cell == 'gru':
            c = tf.compat.v1.nn.rnn_cell.GRUCell(
                num_units=self._layer_width
            )
        elif _cell == 'grublock':
            c = tf.contrib.rnn.GRUBlockCellV2(
                num_units=self._layer_width
            )
        elif _cell == 'lstm':
            c = tf.compat.v1.nn.rnn_cell.LSTMCell(
                num_units=self._layer_width,
                use_peepholes=self._use_peepholes
            )
        elif _cell == 'basiclstm':
            c = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(
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
        output, _ = tf.compat.v1.nn.dynamic_rnn(
            # output, _ = tf.nn.dynamic_rnn(
            c,
            inputs,
            dtype=tf.float32,
            sequence_length=self.seqlen
        )
        output = self.last_relevant(output, self.seqlen)
        print('last time step: {}'.format(output.get_shape()))
        return output

    @staticmethod
    def last_relevant(output, length):
        with tf.compat.v1.name_scope("last_relevant"):
            batch_size = tf.shape(input=output)[0]
            relevant = tf.gather_nd(output, tf.stack(
                [tf.range(batch_size), length-1], axis=1))
            return relevant

    @lazy_property
    def cost(self):
        return tf.reduce_mean(input_tensor=self.xentropy, name="cost")

    @lazy_property
    def optimize(self):
        return tf.compat.v1.train.AdamOptimizer(self._learning_rate,
                                      epsilon=1e-7).minimize(
            self.cost, global_step=tf.compat.v1.train.get_global_step())

    @lazy_property
    def xentropy(self):
        logits = self.logits
        with tf.compat.v1.name_scope("xentropy"):
            return tf.nn.softmax_cross_entropy_with_logits(
                labels=self.target, logits=logits)

    @lazy_property
    def worst(self):
        logits = self.logits
        xentropy = self.xentropy
        with tf.compat.v1.name_scope("worst"):
            bidx = tf.argmax(input=xentropy)
            max_entropy = tf.reduce_max(input_tensor=xentropy)
            shift = len(self._classes)//2
            predict = tf.gather(tf.argmax(input=logits, axis=1), bidx)-shift
            actual = tf.argmax(input=tf.gather(self.target, bidx))-shift
            return bidx, max_entropy, predict, actual

    @lazy_property
    def accuracy(self):
        with tf.compat.v1.name_scope("accuracy"):
            accuracy = tf.equal(
                tf.argmax(input=self.target, axis=1), tf.argmax(input=self.logits, axis=1))
            return tf.reduce_mean(input_tensor=tf.cast(accuracy, tf.float32), name="accuracy")

    @lazy_property
    def one_hot(self):
        logits = self.logits
        size = len(self._classes)
        with tf.compat.v1.name_scope("one_hot"):
            return tf.one_hot(
                tf.argmax(input=logits, axis=1), size, axis=-1)

    @lazy_property
    def precisions(self):
        predictions = self.one_hot
        size = len(self._classes)
        with tf.compat.v1.name_scope("Precisions"):
            ps = []
            ops = []
            for i, c in enumerate(self._classes):
                mask = tf.one_hot([i], size, axis=-1)
                p, op = precision(
                    labels=self.target,
                    predictions=predictions,
                    weights=mask
                )
                tf.compat.v1.summary.scalar("c{}_{}".format(i, c), p*100)
                ps.append(p)
                ops.append(op)
            return ps, ops

    @lazy_property
    def recalls(self):
        predictions = self.one_hot
        size = len(self._classes)
        with tf.compat.v1.name_scope("Recalls"):
            rs = []
            ops = []
            for i, c in enumerate(self._classes):
                mask = tf.one_hot([i], size, axis=-1)
                r, op = recall(
                    labels=self.target,
                    predictions=predictions,
                    weights=mask
                )
                tf.compat.v1.summary.scalar("c{}_{}".format(i, c), r*100)
                rs.append(r)
                ops.append(op)
            return rs, ops

    @lazy_property
    def f_score(self):
        size = len(self._classes)
        mid = size // 2
        ps = self.precisions[0]
        rs = self.recalls[0]
        with tf.compat.v1.name_scope("Fscore"):
            ops = []
            for i, c in enumerate(self._classes):
                b = 2.0
                if i == mid:
                    b = 1.0
                elif i > mid:
                    b = 0.5
                p = ps[i]
                r = rs[i]
                nu = (1.+b**2.) * p * r
                de = (b**2. * p + r)
                op = tf.compat.v1.where(tf.less(de, 1e-7), de, nu/de)
                ops.append(op)
                tf.compat.v1.summary.scalar("c{}_{}".format(i, c), op*100)
            return ops
