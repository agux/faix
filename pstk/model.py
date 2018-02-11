from __future__ import print_function
import functools
import tensorflow as tf
import numpy as np
import math


def primes(n):
    primfac = []
    d = 2
    while d*d <= n:
        while (n % d) == 0:
            primfac.append(d)  # supposing you want multiple factors repeated
            n //= d
        d += 1
    if n > 1:
        primfac.append(n)
    return primfac


def numLayers(feat):
    n = 0
    while feat > 1:
        feat = math.ceil(feat/2)
        n += 1
    return n


def factorize(vspan):
    factors = sorted(primes(vspan))
    if len(factors) < 2:
        raise ValueError('vspan is not factorizable.')
    return tuple(sorted([reduce(lambda x, y: x*y, factors[:-1]), factors[-1]], reverse=True))


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class SecurityGradePredictor:

    def __init__(self, data, target, wsize, training, num_hidden=200, num_layers=2, learning_rate=1e-4):
        self.data = data
        self.target = target
        self.training = training
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self._learning_rate = learning_rate
        self._wsize = wsize
        self.prediction
        self.error
        self.accuracy
        self.optimize

    @lazy_property
    def multi_cnn(self):
        """Model function for CNN."""
        # Add Channel Dimension to Input Layer
        input = tf.expand_dims(self.data, [3])

        step = int(self.data.get_shape()[1])
        feat = int(self.data.get_shape()[2])
        nlayer = numLayers(feat)
        wsize = self._wsize
        print("window size:{} step:{} feat:{} #conv layers: {}".format(
            wsize, step, feat, nlayer))
        filters = max(2, 2 ** (math.ceil(math.log(feat, 2))))
        krange = 3
        drange = 3
        convlayers = np.array([[input for _ in range(drange)]
                               for _ in range(krange)])
        for i in range(nlayer):
            filters *= 2
            uf = math.ceil(filters/(krange*drange))
            for k in range(krange):
                for d in range(drange):
                    conv = tf.layers.conv2d(
                        inputs=convlayers[k][d],
                        filters=uf,
                        dilation_rate=d+1,
                        kernel_size=[k+wsize, k+2],
                        padding="same",
                        activation=tf.nn.elu)
                    pool = tf.layers.max_pooling2d(
                        inputs=conv, pool_size=k+2, strides=[1, 2],
                        padding="same")
                    convlayers[k][d] = pool
                    print("#{} conv:{}\tpool: {}\twide: {}\tdilation: {}".format(
                        i+1, conv.get_shape(), pool.get_shape(), k+2, d+1))
        # Flatten convlayers
        convlayers = convlayers.flatten()
        convlayer = tf.concat([c for c in convlayers], 3)
        print("concat: {}".format(convlayer.get_shape()))
        output_layer = tf.squeeze(convlayer, [2])
        print("cnn output layer: {}".format(output_layer.get_shape()))
        # self.data = output_layer
        return output_layer

    @lazy_property
    def length(self):
        used = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    @lazy_property
    def prediction(self):
        # Recurrent network.
        cells = []
        for _ in range(self._num_layers):
            cell = tf.nn.rnn_cell.GRUCell(
                self._num_hidden)  # Or LSTMCell(num_units), or use ConvLSTMCell?
            # cell = tf.nn.rnn_cell.DropoutWrapper(
            #     cell, output_keep_prob=1.0 - self.dropout)
            cells.append(cell)
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)

        output, _ = tf.nn.dynamic_rnn(
            cell,
            self.multi_cnn,
            dtype=tf.float32,
            sequence_length=self.length,
        )

        last = self._last_relevant(output, self.length)
        # weight, bias = self._weight_and_bias(
        #     self._num_hidden, int(self.target.get_shape()[1]))
        # prediction = tf.matmul(last, weight) + bias

        dense = tf.layers.dense(
            inputs=last, units=self._num_hidden * 3, activation=tf.nn.elu)
        dropout = tf.layers.dropout(
            inputs=dense, rate=math.e/10, training=self.training)

        # Logits Layer
        prediction = tf.layers.dense(
            inputs=dropout, units=int(self.target.get_shape()[1]))

        # prediction = self._cnn_layer(self.data,
        #                              self._ksize,
        #                              int(self.target.get_shape()[1]),
        #                              self.training)
        return prediction

    @lazy_property
    def cost(self):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.target, logits=self.prediction))
        # cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
        return cross_entropy

    @lazy_property
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(self._learning_rate)
        return optimizer.minimize(self.cost, global_step=tf.train.get_global_step())

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @lazy_property
    def accuracy(self):
        accuracy = tf.equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(accuracy, tf.float32))
        # return tf.metrics.accuracy(labels=self.target, predictions=self.prediction)

    @staticmethod
    def _cnn_layer(input, ksize, num_class, training):
        """Model function for CNN."""
        # Transform Input Layer to [-1, max_step, state_size, channel]
        input_layer = tf.expand_dims(input, [3])

        # Convolutional Layer & Pooling Layer #1
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=ksize,
            padding="same",
            activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(
            inputs=conv1, pool_size=ksize, strides=2, padding="same")

        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=ksize,
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(
            inputs=conv2, pool_size=ksize, strides=2, padding="same")

        # Dense Layer
        dh = int(input.get_shape()[1])//4
        dw = int(input.get_shape()[2])//4
        pool2_flat = tf.reshape(pool2, [-1, dh * dw * 64])
        dense = tf.layers.dense(
            inputs=pool2_flat, units=2048, activation=tf.nn.relu)
        dropout = tf.layers.dropout(
            inputs=dense, rate=0.5, training=training)

        # Logits Layer
        logits = tf.layers.dense(inputs=dropout, units=num_class)
        return logits

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.1)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

    @staticmethod
    def _last_relevant(output, length):
        batch_size = tf.shape(output)[0]
        max_length = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant
