from __future__ import print_function
import functools
import tensorflow as tf
import numpy as np
import math
from metrics import precision, recall
from model import lazy_property, numLayers, highway, dense_block, stddev


class TcnPredictorV1:
    '''
    Temporal Convolutional Network Predictor
    '''

    def __init__(self, data, target, seqlen, classes, training, dropout,
                 layer_width=200,  learning_rate=1e-3):
        self.data = data
        self.target = target
        self.seqlen = seqlen
        self._layer_width = layer_width
        self._classes = classes
        self._learning_rate = learning_rate
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
        output = self.tcn(self, self.data)
        # layer = self.fcn(self, layer)
        # layer = tf.Print(layer, [layer], "fcn: ", summarize=10)
        # layer = tf.contrib.layers.batch_norm(
        #     inputs=layer,
        #     is_training=self.training,
        #     updates_collections=None
        # )
        # output = tf.layers.dense(
        #     inputs=layer,
        #     units=len(self._classes),
        #     kernel_initializer=tf.truncated_normal_initializer(
        #         stddev=stddev(1.0, int(layer.get_shape()[-1]))),
        #     bias_initializer=tf.constant_initializer(0.1),
        #     activation=tf.nn.selu,
        #     name="output"
        # )
        return output

    @staticmethod
    def tcn(self, inputs):
        # Temporal Convolutional Network
        #TODO: implement me
        return  None


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
