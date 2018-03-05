from __future__ import print_function
import functools
import tensorflow as tf
import numpy as np
import math

from model import lazy_property, numLayers
from cells import EGRUCell, EGRUCell_V1, EGRUCell_V2


class ERnnPredictorV1:

    def __init__(self, data, target, seqlen, height, training, dropout, num_hidden=200, num_layers=2, learning_rate=1e-4):
        self.data = data
        self.target = target
        self.seqlen = seqlen
        self._height = height
        self.training = training
        self.dropout = dropout
        self.training_state = None
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self._learning_rate = learning_rate
        self.prediction
        self.accuracy
        self.optimize

    @lazy_property
    def prediction(self):
        rnn = self.rnn(self, self.data)
        dense = tf.layers.dense(
            inputs=rnn,
            units=self._num_hidden * 3,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            bias_initializer=tf.constant_initializer(0.1),
            activation=tf.nn.relu6)
        dropout = tf.layers.dropout(
            inputs=dense, rate=0.5, training=self.training)
        output = tf.layers.dense(
            inputs=dropout,
            units=int(self.target.get_shape()[1]),
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            bias_initializer=tf.constant_initializer(0.1),
            activation=tf.nn.elu)
        return output

    @staticmethod
    def rnn(self, input):
        # Recurrent network.
        egru = EGRUCell_V1(
            num_units=self._num_hidden,
            shape=[self._height, int(input.get_shape()[2])//self._height],
            kernel=[3, 3],
            kernel_initializer=tf.truncated_normal_initializer(
                stddev=0.01),
            bias_initializer=tf.constant_initializer(0.1)
        )

        cell = tf.nn.rnn_cell.MultiRNNCell([egru] * self._num_layers)

        output, self.training_state = tf.nn.dynamic_rnn(
            cell,
            input,
            dtype=tf.float32,
            sequence_length=self.seqlen,
            initial_state=self.training_state
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
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.target, logits=self.prediction))
        # cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
        return cross_entropy

    @lazy_property
    def optimize(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        optimizer = None
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(
                self.cost, global_step=tf.train.get_global_step())
        return optimizer

    @lazy_property
    def accuracy(self):
        accuracy = tf.equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(accuracy, tf.float32))


def getIndices(x, n):
    # print("x0:{}  x1:{}  batch:{}  n:{}".format(x[0], x[1], batch,n))
    indices = tf.stack([tf.fill([n], x[0]), [
        x[1]-n+i for i in range(n)]], axis=1)
    return indices


class ERnnPredictorV2:

    def __init__(self, data, target, seqlen, height, training, dropout, num_hidden=200, num_layers=2, learning_rate=1e-4):
        self.data = data
        self.target = target
        self.seqlen = seqlen
        self._height = height
        self.training = training
        self.dropout = dropout
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self._num_class = int(target.get_shape()[1])
        self._learning_rate = learning_rate
        self.prediction
        self.accuracy
        self.optimize

    @lazy_property
    def prediction(self):
        nn = self.rnn(self, self.data)
        # cnn = self.cnn(self, rnn)
        with tf.variable_scope("output"):
            # dense = tf.contrib.bayesflow.layers.dense_flipout(
            #     inputs=cnn,
            #     units=self._num_hidden * 4,
            #     activation=tf.nn.elu)
            # output = tf.contrib.bayesflow.layers.dense_flipout(
            #     inputs=dense,
            #     units=self._num_class,
            #     activation=tf.nn.elu)
            # return output
            dense = tf.layers.dense(
                inputs=nn,
                units=self._num_hidden*3,
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=0.01),
                bias_initializer=tf.constant_initializer(0.1),
                activation=tf.nn.relu6)
            # dense = tf.contrib.layers.layer_norm(inputs=dense)
            dense = tf.layers.dropout(
                inputs=dense, rate=0.5, training=self.training)
            output = tf.layers.dense(
                inputs=dense,
                units=int(self.target.get_shape()[1]),
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=0.01),
                bias_initializer=tf.constant_initializer(0.1),
                activation=tf.nn.elu)
            return output

    @staticmethod
    def cnn(self, input):
        with tf.variable_scope("cnn"):
            # Transforms into 2D compatible format [batch, step, feature, channel]
            convlayer = tf.expand_dims(input, 3)
            height = int(input.get_shape()[1])
            width = int(input.get_shape()[2])
            nlayer = min(5, numLayers(height, width))
            filters = max(
                16, 2 ** (math.floor(math.log(self._num_hidden//nlayer))))
            for i in range(nlayer):
                filters = min(filters*2, self._num_hidden*2)
                convlayer = self.conv2d(
                    convlayer, int(filters), i, tf.nn.relu6)
            convlayer = tf.layers.flatten(convlayer)
            # convlayer = tf.contrib.layers.layer_norm(inputs=convlayer)
            # convlayer = tf.layers.dense(
            #     inputs=convlayer,
            #     units=math.ceil(
            #         math.sqrt(float(int(convlayer.get_shape()[1])))),
            #     kernel_initializer=tf.truncated_normal_initializer(
            #         stddev=0.01),
            #     bias_initializer=tf.constant_initializer(0.1)
            # )
            return convlayer

    @staticmethod
    def conv2d(input, filters, i, activation=None):
        with tf.variable_scope("conv{}".format(i)):
            h = int(input.get_shape()[1])
            w = int(input.get_shape()[2])
            conv = tf.layers.conv2d(
                inputs=input,
                filters=filters,
                kernel_size=2,
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=0.01),
                bias_initializer=tf.constant_initializer(0.1),
                activation=activation,
                padding="SAME")
            h_stride = 2 if (h > 2 or w == 2) else 1
            w_stride = 2 if (w > 2 or h == 2) else 1
            pool = tf.layers.max_pooling2d(
                inputs=conv, pool_size=2, strides=[h_stride, w_stride],
                padding="SAME")
            pool = tf.contrib.layers.layer_norm(inputs=pool)
            print("conv{}: {}".format(i, pool.get_shape()))
            return pool

    @staticmethod
    def rnn(self, input):
        # Recurrent network.
        egru = EGRUCell_V2(
            num_units=self._num_hidden,
            shape=[self._height, int(input.get_shape()[2])//self._height],
            kernel=[3, 3],
            kernel_initializer=tf.truncated_normal_initializer(
                stddev=0.01),
            bias_initializer=tf.constant_initializer(0.1),
            training=self.training
        )
        cell = tf.nn.rnn_cell.MultiRNNCell([egru] * self._num_layers)
        output, _ = tf.nn.dynamic_rnn(
            cell,
            input,
            dtype=tf.float32,
            sequence_length=self.seqlen
        )
        # output = self.last_relevant(output, self.seqlen)
        output = self.last_n(output, self.seqlen, 1)
        output = tf.layers.flatten(output)
        return output

    @staticmethod
    def last_n(output, length, n):
        '''
        input:  [batch, max_step, feature]
        output: [batch, n, feature]
        where n is the last n features in max_step.
        '''
        with tf.variable_scope("last_n"):
            batch = tf.shape(output)[0]
            indices = tf.map_fn(lambda x: getIndices(x, n),
                                tf.stack([tf.range(batch), length], axis=1))
            return tf.gather_nd(output, indices)

    @staticmethod
    def last_relevant(output, length):
        with tf.variable_scope("last_relevant"):
            batch_size = tf.shape(output)[0]
            relevant = tf.gather_nd(output, tf.stack(
                [tf.range(batch_size), length-1], axis=1))
            return relevant

    # def reduce_sum(self, input):
    #     print("reduce_sum:{}".format(input.get_shape()))
    #     # input=tf.reshape(input, [None, self._num_class])
    #     return

    def merge(self, items):
        with tf.variable_scope("merge"):
            items = [tf.convert_to_tensor(i, dtype=tf.float32) for i in items]
            items = [tf.reshape(i, shape=[-1]) for i in items]
            items = tf.concat(items, axis=0)
            return items

    # @lazy_property
    # def kl_loss(self):
    #     with tf.variable_scope("kl_loss"):
    #         regl_loss=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    #         losses=[tf.reduce_mean(x) for x in regl_loss]
    #         return tf.reduce_mean(losses)

    # @lazy_property
    # def cost(self):
    #     cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    #         labels=self.target, logits=self.prediction))
    #     loss=tf.reduce_mean([cross_entropy, self.kl_loss])
    #     return loss

    @lazy_property
    def cost(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.target, logits=self.prediction, name="xentropy")
        # regl_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # regl_loss = self.merge(regl_loss)
        # loss = tf.reduce_logsumexp(
        #     tf.concat([cross_entropy, regl_loss], axis=0))
        # loss = tf.reduce_sum(tf.concat([cross_entropy, regl_loss], axis=0))
        loss = tf.reduce_mean(cross_entropy)
        return loss

    @lazy_property
    def optimize(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            return tf.train.AdamOptimizer(self._learning_rate).minimize(
                self.cost, global_step=tf.train.get_global_step())

    @lazy_property
    def accuracy(self):
        accuracy = tf.equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(accuracy, tf.float32), name="accuracy")
