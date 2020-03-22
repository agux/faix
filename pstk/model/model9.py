from __future__ import print_function
import functools
import tensorflow as tf
import numpy as np
import math
from metrics import precision, recall
from model import lazy_property, numLayers, highway, dense_block, stddev
from cells import EGRUCell, EGRUCell_V1, EGRUCell_V2, LayerNormGRUCell, LayerNormNASCell, DenseCellWrapper, AlphaDropoutWrapper


class DRnnPredictorV5:
    '''
    Deep RNN + FCN predictor, with densenet connection, self-normalization
    '''

    def __init__(self, data, target, seqlen, classes, training, dropout,
                 layer_width=200, num_rnn_layers=10, rnn_layer_size=2, num_fcn_layers=10, size_decay=0.3, learning_rate=1e-3):
        self.data = data
        self.target = target
        self.seqlen = seqlen
        self._layer_width = layer_width
        self._num_rnn_layers = num_rnn_layers
        self._rnn_layer_size = rnn_layer_size
        self._num_fcn_layers = num_fcn_layers
        self._classes = classes
        self._learning_rate = learning_rate
        self._size_decay = size_decay
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
        layer = self.fcn(self, layer)
        layer = tf.compat.v1.Print(layer,[layer],"fcn: ",summarize=10)
        layer = tf.contrib.layers.batch_norm(
            inputs=layer,
            is_training=self.training,
            updates_collections=None
        )
        output = tf.compat.v1.layers.dense(
            inputs=layer,
            units=len(self._classes),
            kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                stddev=stddev(1.0, int(layer.get_shape()[-1]))),
            bias_initializer=tf.compat.v1.constant_initializer(0.1),
            activation=tf.nn.selu,
            name="output"
        )
        return output

    @staticmethod
    def fcn(self, input):
        block = input
        with tf.compat.v1.variable_scope("fcn"):
            p = int(round(self._num_fcn_layers ** 0.5))
            for i in range(self._num_fcn_layers):
                with tf.compat.v1.variable_scope("dense_block_{}".format(i+1)):
                    block = tf.contrib.layers.batch_norm(
                        inputs=block,
                        is_training=self.training,
                        updates_collections=None
                    )
                    if i > 0 and i % p == 0:
                        block = tf.nn.selu(block)
                        size = int(block.get_shape()[-1])
                        new_size = int(round(size*self._size_decay))
                        block = tf.compat.v1.layers.dense(
                            inputs=block,
                            units=new_size,
                            kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                                stddev=stddev(1.0, size)),
                            bias_initializer=tf.compat.v1.constant_initializer(0.1)
                        )
                        block = tf.contrib.nn.alpha_dropout(
                            block, 1.0 - self.dropout)
                        print("fcn layer_{} decayed size:{}".format(i, new_size))
                    else:
                        block = dense_block(block, self._layer_width)
                        print("fcn layer_{} size:{}".format(
                            i, block.get_shape()[-1]))
            return block

    @staticmethod
    def rnn(self, inputs):
        # Deep Recurrent network.
        feat_size = int(inputs.get_shape()[-1])
        p = int(round(self._num_rnn_layers ** 0.5))
        output_size = feat_size
        block = inputs
        with tf.compat.v1.variable_scope("dense_rnn"):
            for i in range(self._num_rnn_layers):
                with tf.compat.v1.variable_scope("rnn_{}".format(i+1)):
                    if i > 0:
                        block = tf.contrib.layers.batch_norm(
                            inputs=block,
                            is_training=self.training,
                            updates_collections=None
                        )
                        block = tf.compat.v1.Print(
                            block, [block], "{}_batch_norm: ".format(i),summarize=10)
                        # block = tf.nn.selu(block, name="selu")
                    if i == 0 or i % p != 0:
                        output_size += self._layer_width
                        c = DenseCellWrapper(LayerNormGRUCell(
                            num_units=self._layer_width,
                            kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                                stddev=stddev(1.0, feat_size)),
                            bias_initializer=tf.compat.v1.constant_initializer(0.1)
                        ), output_size=output_size)
                        block, _ = self.rnn_block(block, c, self.seqlen)
                        block = tf.compat.v1.Print(
                            block, [block], "{}_rnn_block: ".format(i),summarize=10)
                        print("rnn layer_{} size:{}".format(i, output_size))
                    else:
                        # bottleneck
                        output_size = int(
                            round(output_size * self._size_decay))
                        c = LayerNormNASCell(
                            num_units=output_size,
                            use_biases=True
                        )
                        block, _ = self.rnn_block(block, c, self.seqlen)
                        block = tf.compat.v1.Print(
                            block, [block], "{}_bottleneck: ".format(i),summarize=10)
                        block = tf.contrib.nn.alpha_dropout(
                            block, 1.0 - self.dropout)
                        block = tf.compat.v1.Print(
                            block, [block], "{}_bottleneck_dropout: ".format(i),summarize=10)
                        print("rnn layer_{} decayed size:{}".format(i, output_size))
            return self.last_relevant(block, self.seqlen)

    @staticmethod
    def rnn_block(inputs, cell, seqlen):
        return tf.compat.v1.nn.dynamic_rnn(
            cell,
            inputs,
            dtype=tf.float32,
            sequence_length=seqlen
        )

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
