from __future__ import print_function
import functools
import tensorflow as tf
import numpy as np
import math
from metrics import precision, recall
from model import lazy_property, residual, stddev
from cells import LayerNormNASCell, LayerNormGRUCell


class DRnnPredictorV6:
    '''
    Deep Residual RNN+FCN Predictor
    '''

    def __init__(self, data, target, seqlen, classes, training, dropout,
                 layer_width=200, rnn_layers=16, fcn_layers=16, learning_rate=1e-3):
        self.data = data
        self.target = target
        self.seqlen = seqlen
        self._layer_width = layer_width
        self._classes = classes
        self._learning_rate = learning_rate
        self._rnn_layers = rnn_layers
        self._fcn_layers = fcn_layers
        self.training = training
        self.dropout = dropout
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
        # layer = self.fcn(self, layer)
        # layer = tf.Print(layer, [layer], "fcn: ", summarize=10)
        # layer = tf.contrib.layers.batch_norm(
        #     inputs=layer,
        #     is_training=self.training,
        #     updates_collections=None
        # )
        layer = tf.contrib.nn.alpha_dropout(
            layer, 1.0 - self.dropout)
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
    def fcn(self, inputs):
        fc = inputs
        # p = int(round(self._fcn_layers ** 0.5))
        with tf.variable_scope("fcn"):
            fc = tf.contrib.layers.batch_norm(
                inputs=fc,
                is_training=self.training,
                updates_collections=None
            )
            for i in range(self._fcn_layers):
                activation = None
                if i == self._fcn_layers-1:
                    activation = tf.nn.selu
                with tf.variable_scope("residual_{}".format(i)):
                    fc = residual(
                        x=fc,
                        activation=activation
                    )
                # if i > 0 and i % p == 0:
                #     tf.contrib.nn.alpha_dropout(
                #         block, 1.0 - self.dropout)
            return fc

    @staticmethod
    def rnn(self, inputs):
        # Deep Residual RNN
        cells = []
        feat_size = int(inputs.get_shape()[-1])
        # p = int(round(self._rnn_layers ** 0.5))
        if feat_size != self._layer_width:
            # dimensionality adaptation
            c = tf.nn.rnn_cell.GRUCell(
                num_units=self._layer_width,
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=stddev(1.0, self._layer_width)),
                bias_initializer=tf.constant_initializer(0.1)
            )
            # c = LayerNormNASCell(
            #     num_units=self._layer_width,
            #     use_biases=True
            # )
            cells.append(c)
        for _ in range(self._rnn_layers):
            # if i == 0 or i % p != 0:
            c = tf.contrib.rnn.ResidualWrapper(tf.contrib.rnn.LayerNormBasicLSTMCell(
                num_units=self._layer_width
                # kernel_initializer=tf.truncated_normal_initializer(
                #     stddev=stddev(1.0, self._layer_width)),
                # bias_initializer=tf.constant_initializer(0.1)
                # layer_norm=True
            ))
            # else:
            # c = tf.contrib.rnn.ResidualWrapper(tf.contrib.rnn.NASCell(
            #     num_units=self._layer_width,
            #     use_biases=True
            #     # layer_norm=True
            # ))
            cells.append(c)
        # Stack layers of cell
        mc = tf.nn.rnn_cell.MultiRNNCell(cells)
        output, _ = tf.nn.dynamic_rnn(
            mc,
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
