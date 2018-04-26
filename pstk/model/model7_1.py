from __future__ import print_function
import functools
import tensorflow as tf
import numpy as np
import math
from metrics import precision, recall
from model import lazy_property, residual, stddev
from cells import LayerNormNASCell, LayerNormGRUCell


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

    def __init__(self, data, target, seqlen, classes, cell, use_peepholes=False, time_shifts=1,
                 layer_width=200, learning_rate=1e-3):
        self.data = data
        self.target = target
        self.seqlen = seqlen
        self._layer_width = layer_width
        self._classes = classes
        self._learning_rate = learning_rate
        self._cell = cell
        self._use_peepholes = use_peepholes
        self._time_shifts = time_shifts
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
        output = tf.layers.dense(
            inputs=layer,
            units=len(self._classes),
            kernel_initializer=tf.truncated_normal_initializer(
                stddev=stddev(1.0, int(layer.get_shape()[-1]))),
            bias_initializer=tf.constant_initializer(0.1),
            activation=tf.nn.relu6,
            name="output"
        )
        return output

    @staticmethod
    def rnn(self, inputs):
        c = None
        _cell = self._cell.lower()
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
                number_of_groups=self._time_shifts
            )
        elif _cell == 'gridlstm':
            c = tf.contrib.rnn.GridLSTMCell(
                num_units=self._layer_width,
                use_peephole=self._use_peepholes
                # share_time_frequency_weights=True,
                # num_unit_shards=4,
                #feature_size = feat_size,
                # num_frequency_blocks = ?
            )
        elif _cell == 'lstmblock':
            c = tf.contrib.rnn.LSTMBlockCell(
                num_units=self._layer_width
                # use_peephole=True
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
        output, _ = tf.nn.dynamic_rnn(
            c,
            inputs,
            dtype=tf.float32,
            sequence_length=self.seqlen
        )
        return self.last_relevant(output, self.seqlen)

    @staticmethod
    def last_relevant(output, length):
        with tf.name_scope("last_relevant"):
            batch_size = tf.shape(output)[0]
            relevant = tf.gather_nd(output, tf.stack(
                [tf.range(batch_size), length-1], axis=1))
            return relevant

    @lazy_property
    def cost(self):
        return tf.reduce_mean(self.xentropy, name="cost")

    @lazy_property
    def optimize(self):
        return tf.train.AdamOptimizer(self._learning_rate,
                                      epsilon=1e-7).minimize(
            self.cost, global_step=tf.train.get_global_step())

    @lazy_property
    def xentropy(self):
        logits = self.logits
        with tf.name_scope("xentropy"):
            return tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.target, logits=logits)

    @lazy_property
    def worst(self):
        logits = self.logits
        xentropy = self.xentropy
        with tf.name_scope("worst"):
            bidx = tf.argmax(xentropy)
            max_entropy = tf.reduce_max(xentropy)
            shift = len(self._classes)//2
            predict = tf.gather(tf.argmax(logits, 1), bidx)-shift
            actual = tf.argmax(tf.gather(self.target, bidx))-shift
            return bidx, max_entropy, predict, actual

    @lazy_property
    def accuracy(self):
        with tf.name_scope("accuracy"):
            accuracy = tf.equal(
                tf.argmax(self.target, 1), tf.argmax(self.logits, 1))
            return tf.reduce_mean(tf.cast(accuracy, tf.float32), name="accuracy")

    @lazy_property
    def one_hot(self):
        logits = self.logits
        size = len(self._classes)
        with tf.name_scope("one_hot"):
            return tf.one_hot(
                tf.argmax(logits, 1), size, axis=-1)

    @lazy_property
    def precisions(self):
        predictions = self.one_hot
        size = len(self._classes)
        with tf.name_scope("Precisions"):
            ps = []
            ops = []
            for i, c in enumerate(self._classes):
                mask = tf.one_hot([i], size, axis=-1)
                p, op = precision(
                    labels=self.target,
                    predictions=predictions,
                    weights=mask
                )
                tf.summary.scalar("c{}_{}".format(i, c), p*100)
                ps.append(p)
                ops.append(op)
            return ps, ops

    @lazy_property
    def recalls(self):
        predictions = self.one_hot
        size = len(self._classes)
        with tf.name_scope("Recalls"):
            rs = []
            ops = []
            for i, c in enumerate(self._classes):
                mask = tf.one_hot([i], size, axis=-1)
                r, op = recall(
                    labels=self.target,
                    predictions=predictions,
                    weights=mask
                )
                tf.summary.scalar("c{}_{}".format(i, c), r*100)
                rs.append(r)
                ops.append(op)
            return rs, ops

    @lazy_property
    def f_score(self):
        size = len(self._classes)
        mid = size // 2
        ps = self.precisions[0]
        rs = self.recalls[0]
        with tf.name_scope("Fscore"):
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
                op = tf.where(tf.less(de, 1e-7), de, nu/de)
                ops.append(op)
                tf.summary.scalar("c{}_{}".format(i, c), op*100)
            return ops
