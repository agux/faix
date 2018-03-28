from __future__ import print_function
import functools
import tensorflow as tf
import numpy as np
import math
from metrics import precision, recall
from model import lazy_property, numLayers
from cells import EGRUCell, EGRUCell_V1, EGRUCell_V2


class MRnnPredictorV2:

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
        layer = self.rnn(self, self.data)
        # ln = tf.contrib.layers.layer_norm(inputs=rnn)
        layer = self.dnn(self, layer)
        output = tf.layers.dense(
            inputs=layer,
            units=self._num_class,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            bias_initializer=tf.constant_initializer(0.1),
            activation=tf.nn.elu
        )
        return output

    @staticmethod
    def dnn(self, input):
        with tf.variable_scope("dnn"):
            dense = tf.layers.dense(
                inputs=input,
                units=self._num_hidden,
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=0.01),
                bias_initializer=tf.constant_initializer(0.1)
                # activation=tf.nn.elu
            )
            dense = tf.layers.dense(
                inputs=dense,
                units=self._num_hidden//2,
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=0.01),
                bias_initializer=tf.constant_initializer(0.1),
                # activation=tf.nn.elu
            )
            dense = tf.layers.dense(
                inputs=dense,
                units=self._num_hidden//4,
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=0.01),
                bias_initializer=tf.constant_initializer(0.1),
                activation=tf.nn.relu6
            )
            dropout = tf.layers.dropout(
                inputs=dense, rate=0.5, training=self.training)
            return dropout

    @staticmethod
    def rnn(self, input):
        # Recurrent network.
        c1 = tf.nn.rnn_cell.GRUCell(
            num_units=self._num_hidden,
            kernel_initializer=tf.truncated_normal_initializer(
                stddev=0.01),
            bias_initializer=tf.constant_initializer(0.1)
        )
        c1 = tf.nn.rnn_cell.DropoutWrapper(
            cell=c1,
            input_keep_prob=(1.0-self.dropout)
        )
        c2 = tf.nn.rnn_cell.GRUCell(
            num_units=self._num_hidden//2,
            kernel_initializer=tf.truncated_normal_initializer(
                stddev=0.01),
            bias_initializer=tf.constant_initializer(0.1)
        )

        mc = tf.nn.rnn_cell.MultiRNNCell([c1, c2] * self._num_layers)
        mc = tf.nn.rnn_cell.DropoutWrapper(
            cell=mc,
            output_keep_prob=(1.0-self.dropout)
        )
        output, _ = tf.nn.dynamic_rnn(
            mc,
            input,
            dtype=tf.float32,
            sequence_length=self.seqlen
        )

        return self.last_relevant(output, self.seqlen)

    @staticmethod
    def last_relevant(output, length):
        batch_size = tf.shape(output)[0]
        relevant = tf.gather_nd(output, tf.stack(
            [tf.range(batch_size), length-1], axis=1))
        return relevant

    @lazy_property
    def cost(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.target, logits=self.prediction, name="xentropy")
        loss = tf.reduce_mean(cross_entropy)
        return loss

    @lazy_property
    def optimize(self):
        return tf.train.AdamOptimizer(self._learning_rate).minimize(
            self.cost, global_step=tf.train.get_global_step())

    @lazy_property
    def accuracy(self):
        accuracy = tf.equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(accuracy, tf.float32), name="accuracy")


class MRnnPredictorV3:

    def __init__(self, data, target, seqlen, classes, training, dropout, num_hidden=200, num_layers=1, learning_rate=1e-4):
        self.data = data
        self.target = target
        self.seqlen = seqlen
        self.training = training
        self.dropout = dropout
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self._classes = classes
        self._learning_rate = learning_rate
        self.keep_prob
        self.precisions
        self.recalls
        self.prediction
        self.accuracy
        self.optimize
        self.cost
        self.one_hot

    @lazy_property
    def prediction(self):
        layer = self.rnn(self, self.data)
        # ln = tf.contrib.layers.layer_norm(inputs=rnn)
        layer = self.dnn(self, layer)
        output = tf.layers.dense(
            inputs=layer,
            units=len(self._classes),
            kernel_initializer=tf.truncated_normal_initializer(
                stddev=0.01),
            bias_initializer=tf.constant_initializer(0.1),
            activation=tf.nn.elu,
            name="output"
        )
        return output

    @staticmethod
    def dnn(self, input):
        with tf.variable_scope("DNN"):
            dense = tf.layers.dense(
                inputs=input,
                units=self._num_hidden,
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=0.01),
                bias_initializer=tf.constant_initializer(0.1)
                # activation=tf.nn.elu
            )
            dense = tf.layers.dense(
                inputs=dense,
                units=self._num_hidden//2,
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=0.01),
                bias_initializer=tf.constant_initializer(0.1),
                # activation=tf.nn.elu
            )
            dense = tf.layers.dense(
                inputs=dense,
                units=self._num_hidden//4,
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=0.01),
                bias_initializer=tf.constant_initializer(0.1),
                activation=tf.nn.relu6
            )
            dropout = tf.layers.dropout(
                inputs=dense, rate=0.6, training=self.training)
            return dropout

    @staticmethod
    def rnn(self, input):
        # Recurrent network.
        cells = []
        for i in range(self._num_layers):
            c = tf.nn.rnn_cell.GRUCell(
                num_units=self._num_hidden//(2**i),
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=0.01),
                bias_initializer=tf.constant_initializer(0.1)
            )
            if i > 0:
                c = tf.nn.rnn_cell.DropoutWrapper(
                    cell=c,
                    input_keep_prob=self.keep_prob
                )
            cells.append(c)
        mc = tf.nn.rnn_cell.MultiRNNCell(cells)
        mc = tf.nn.rnn_cell.DropoutWrapper(
            cell=mc,
            output_keep_prob=self.keep_prob
        )
        output, _ = tf.nn.dynamic_rnn(
            mc,
            input,
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
    def keep_prob(self):
        with tf.name_scope("keep_prob"):
            return 1.0-self.dropout

    @lazy_property
    def cost(self):
        prediction = self.prediction
        with tf.name_scope("cost"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.target, logits=prediction)
            loss = tf.reduce_mean(cross_entropy)
            return loss

    @lazy_property
    def optimize(self):
        return tf.train.AdamOptimizer(self._learning_rate).minimize(
            self.cost, global_step=tf.train.get_global_step())

    @lazy_property
    def accuracy(self):
        with tf.name_scope("accuracy"):
            accuracy = tf.equal(
                tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
            return tf.reduce_mean(tf.cast(accuracy, tf.float32), name="accuracy")

    @lazy_property
    def one_hot(self):
        prediction = self.prediction
        size = len(self._classes)
        with tf.name_scope("one_hot"):
            return tf.one_hot(
                tf.argmax(prediction, 1), size, axis=-1)

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
                    weights=mask,
                    updates_collections=tf.GraphKeys.UPDATE_OPS
                )
                tf.summary.scalar("c_{}".format(c), p*100)
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
                    weights=mask,
                    updates_collections=tf.GraphKeys.UPDATE_OPS
                )
                tf.summary.scalar("c_{}".format(c), r*100)
                rs.append(r)
                ops.append(op)
            return rs, ops


class MRnnPredictorV4:

    def __init__(self, data, target, seqlen, classes, dropout, num_hidden=200, num_layers=1, learning_rate=1e-4):
        self.data = data
        self.target = target
        self.seqlen = seqlen
        self.dropout = dropout
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self._classes = classes
        self._learning_rate = learning_rate
        self.keep_prob
        self.precisions
        self.recalls
        self.f_score
        self.prediction
        self.accuracy
        self.optimize
        self.cost
        self.one_hot

    @lazy_property
    def prediction(self):
        layer = self.rnn(self, self.data)
        # ln = tf.contrib.layers.layer_norm(inputs=rnn)
        layer = self.dnn(self, layer)
        output = tf.layers.dense(
            inputs=layer,
            units=len(self._classes),
            kernel_initializer=tf.truncated_normal_initializer(
                stddev=0.01),
            bias_initializer=tf.constant_initializer(0.1),
            activation=tf.nn.elu,
            name="output"
        )
        return output

    @staticmethod
    def dnn(self, input):
        with tf.variable_scope("DNN"):
            dense = tf.layers.dense(
                inputs=input,
                units=self._num_hidden,
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=0.01),
                bias_initializer=tf.constant_initializer(0.1)
                # activation=tf.nn.elu
            )
            dense = tf.layers.dense(
                inputs=dense,
                units=self._num_hidden//2,
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=0.01),
                bias_initializer=tf.constant_initializer(0.1),
                # activation=tf.nn.elu
            )
            dense = tf.layers.dense(
                inputs=dense,
                units=self._num_hidden//4,
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=0.01),
                bias_initializer=tf.constant_initializer(0.1),
                activation=tf.nn.relu6
            )
            dropout = tf.nn.dropout(
                x=dense, keep_prob=self.keep_prob, name="dropout")
            return dropout

    @staticmethod
    def rnn(self, input):
        # Recurrent network.
        cells = []
        for i in range(self._num_layers):
            c = tf.nn.rnn_cell.GRUCell(
                num_units=self._num_hidden//(2**i),
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=0.01),
                bias_initializer=tf.constant_initializer(0.1)
            )
            if i > 0:
                c = tf.nn.rnn_cell.DropoutWrapper(
                    cell=c,
                    input_keep_prob=self.keep_prob
                )
            cells.append(c)
        mc = tf.nn.rnn_cell.MultiRNNCell(cells)
        mc = tf.nn.rnn_cell.DropoutWrapper(
            cell=mc,
            output_keep_prob=self.keep_prob
        )
        output, _ = tf.nn.dynamic_rnn(
            mc,
            input,
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
    def keep_prob(self):
        with tf.name_scope("keep_prob"):
            return 1.0-self.dropout

    @lazy_property
    def cost(self):
        prediction = self.prediction
        with tf.name_scope("cost"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.target, logits=prediction)
            loss = tf.reduce_mean(cross_entropy)
            return loss

    @lazy_property
    def optimize(self):
        return tf.train.AdamOptimizer(self._learning_rate).minimize(
            self.cost, global_step=tf.train.get_global_step())

    @lazy_property
    def accuracy(self):
        with tf.name_scope("accuracy"):
            accuracy = tf.equal(
                tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
            return tf.reduce_mean(tf.cast(accuracy, tf.float32), name="accuracy")

    @lazy_property
    def one_hot(self):
        prediction = self.prediction
        size = len(self._classes)
        with tf.name_scope("one_hot"):
            return tf.one_hot(
                tf.argmax(prediction, 1), size, axis=-1)

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
                    weights=mask,
                    updates_collections=tf.GraphKeys.UPDATE_OPS
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
                    weights=mask,
                    updates_collections=tf.GraphKeys.UPDATE_OPS
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


class MRnnPredictorV5:

    def __init__(self, data, target, seqlen, classes, dropout, num_hidden=200, num_layers=1, learning_rate=1e-4):
        self.data = data
        self.target = target
        self.seqlen = seqlen
        self.dropout = dropout
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self._classes = classes
        self._learning_rate = learning_rate
        self.keep_prob
        self.precisions
        self.recalls
        self.f_score
        self.prediction
        self.accuracy
        self.optimize
        self.cost
        self.one_hot

    @lazy_property
    def prediction(self):
        layer = self.rnn(self, self.data)
        # ln = tf.contrib.layers.layer_norm(inputs=rnn)
        # layer = self.dnn(self, layer)
        output = tf.layers.dense(
            inputs=layer,
            units=len(self._classes),
            kernel_initializer=tf.truncated_normal_initializer(
                stddev=0.01),
            bias_initializer=tf.constant_initializer(0.1),
            activation=tf.nn.elu,
            name="output"
        )
        return output

    @staticmethod
    def rnn(self, input):
        # Recurrent network.
        cells = []
        for i in range(self._num_layers):
            c = tf.nn.rnn_cell.GRUCell(
                num_units=self._num_hidden//(2**i),
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=0.01),
                bias_initializer=tf.constant_initializer(0.1)
            )
            if i > 0:
                c = tf.nn.rnn_cell.DropoutWrapper(
                    cell=c,
                    input_keep_prob=self.keep_prob
                )
            cells.append(c)
        mc = tf.nn.rnn_cell.MultiRNNCell(cells)
        # mc = tf.nn.rnn_cell.DropoutWrapper(
        #     cell=mc,
        #     output_keep_prob=self.keep_prob
        # )
        output, _ = tf.nn.dynamic_rnn(
            mc,
            input,
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
    def keep_prob(self):
        with tf.name_scope("keep_prob"):
            return 1.0-self.dropout

    @lazy_property
    def cost(self):
        prediction = self.prediction
        with tf.name_scope("cost"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.target, logits=prediction)
            loss = tf.reduce_mean(cross_entropy)
            return loss

    @lazy_property
    def optimize(self):
        return tf.train.AdamOptimizer(self._learning_rate).minimize(
            self.cost, global_step=tf.train.get_global_step())

    @lazy_property
    def accuracy(self):
        with tf.name_scope("accuracy"):
            accuracy = tf.equal(
                tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
            return tf.reduce_mean(tf.cast(accuracy, tf.float32), name="accuracy")

    @lazy_property
    def one_hot(self):
        prediction = self.prediction
        size = len(self._classes)
        with tf.name_scope("one_hot"):
            return tf.one_hot(
                tf.argmax(prediction, 1), size, axis=-1)

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
                    weights=mask,
                    updates_collections=tf.GraphKeys.UPDATE_OPS
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
                    weights=mask,
                    updates_collections=tf.GraphKeys.UPDATE_OPS
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