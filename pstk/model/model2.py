from __future__ import print_function
import functools
import tensorflow as tf
import numpy as np
import math
from model import numLayers


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


def factorize(feature_size):
    factors = sorted(primes(feature_size))
    if len(factors) < 2:
        raise ValueError('feature size is not factorizable.')
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


def length(data):
    # with tf.variable_scope("rnn_length"):    #FIXME no scope?
    used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


def fusedBN(input, scale, offset, mean, variance, training):
    return tf.nn.fused_batch_norm(
        x=input, scale=scale, offset=offset, mean=mean, variance=variance, is_training=training)


def last_relevant(output, length):
    # with tf.variable_scope("rnn_last"):       #FIXME no scope?
    batch_size = tf.shape(output)[0]
    max_length = int(output.get_shape()[1])
    output_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, output_size])
    relevant = tf.gather(flat, index)
    return relevant


def conv2d(input, filters, seq):
    conv = tf.layers.conv2d(
        # name="conv_lv{}".format(seq),    #FIXME perhaps no name?
        inputs=input,
        filters=filters,
        kernel_size=2,
        kernel_initializer=tf.truncated_normal_initializer(
            stddev=0.01),
        bias_initializer=tf.constant_initializer(0.1),
        padding="same",
        activation=tf.nn.elu)  # FIXME or perhaps relu6??
    h_stride = 2 if int(conv.get_shape()[1]) >= 2 else 1
    w_stride = 2 if int(conv.get_shape()[2]) >= 2 else 1
    pool = tf.layers.max_pooling2d(
        # name="pool_lv{}".format(seq),    #FIXME perhaps no name?
        inputs=conv, pool_size=2, strides=[h_stride, w_stride],
        padding="same")
    # can't use tf.nn.batch_normalization in a mapped function
    print("#{} conv:{} pool: {}".format(
        seq+1, conv.get_shape(), pool.get_shape()))
    return pool


class CRnnPredictorV1:

    def __init__(self, data, target, height, training, dropout, num_hidden=200, num_layers=2, learning_rate=1e-4):
        self.data = data
        self.target = target
        self.training = training
        self.dropout = dropout
        self.training_state = None
        self._height = height
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self._learning_rate = learning_rate
        self.prediction
        self.accuracy
        self.optimize

    @lazy_property
    def prediction(self):
        print("data shape:{}".format(self.data.get_shape()))
        cnn = tf.map_fn(lambda x: self.cnn2d(
            x, self.training, self._height), self.data)
        print("cnn2d: {}".format(cnn.get_shape()))
        # caught "InvalidArgumentError: Retval[0] does not have value" here
        # cnn2d = tf.layers.batch_normalization(
        #     cnn2d,
        #     beta_initializer=tf.truncated_normal_initializer(0.01),
        #     # gamma_initializer=tf.truncated_normal_initializer(0.01),
        #     moving_mean_initializer=tf.truncated_normal_initializer(0.01),
        #     # moving_variance_initializer=tf.truncated_normal_initializer(0.01),
        #     training=self.training
        # )
        rnn = self.rnn(self, cnn)
        dense = tf.layers.dense(
            inputs=rnn,
            units=self._num_hidden * 3,  # FIXME fallback to 3 * hidden size?
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            bias_initializer=tf.constant_initializer(0.1),
            activation=tf.nn.elu)  # FIXME sure elu?
        dropout = tf.layers.dropout(
            inputs=dense, rate=0.5, training=self.training)
        output = tf.layers.dense(
            inputs=dropout,
            units=int(self.target.get_shape()[1]),
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            bias_initializer=tf.constant_initializer(0.1),
            activation=tf.nn.relu6)  # FIXME fall back to relu6?
        return output

    @staticmethod
    def rnn(self, input):
        # Recurrent network.
        cells = []
        state_size = self._num_hidden  # FIXME fallback to 128
        for _ in range(self._num_layers):
            # Or LSTMCell(num_units), or use ConvLSTMCell?
            cell = tf.nn.rnn_cell.GRUCell(
                state_size,
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=0.01),
                bias_initializer=tf.constant_initializer(0.1))
            # activation=None)  # FIXME fall back to None?
            cells.append(cell)
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        _length = length(input)
        output, self.training_state = tf.nn.dynamic_rnn(
            cell,
            input,
            dtype=tf.float32,
            sequence_length=_length,
            initial_state=self.training_state
        )

        last = last_relevant(output, _length)
        return last

    @staticmethod
    def cnn2d(input, training, height):
        """
        Model function for CNN.
        accepts input shape: [step_size, time_shift*features]
        transformed to: [step_size, time_shift(height), features(width), channel]
        """
        # with tf.variable_scope("conv2d_parent"):    #FIXME no scope?
        print("shape of cnn input: {}".format(input.get_shape()))
        width = int(input.get_shape()[1])//height
        input2d = tf.reshape(input, [-1, height, width, 1])
        # Transforms into 2D compatible format [batch(step), height, width, channel]
        print(
            "input transformed to 2D pre-conv shape: {}".format(input2d.get_shape()))
        nlayer = numLayers(height, width)
        print("height:{} width:{} #conv layers: {}".format(
            height, width, nlayer))
        filters = max(
            2, 2 ** (math.ceil(math.log(max(height, width), 2))))
        # krange = 3
        # drange = 3
        convlayer = input2d

        for i in range(nlayer):
            filters *= 2
            convlayer = conv2d(convlayer, filters, i)

        print("final conv2d: {}".format(convlayer.get_shape()))
        convlayer = tf.squeeze(convlayer, [1, 2])
        print("squeeze: {}".format(convlayer.get_shape()))
        dense = tf.layers.dense(
            # name="cnn2d_dense",     #FIXME no name?
            inputs=convlayer,
            units=convlayer.get_shape()[1]*2,
            kernel_initializer=tf.truncated_normal_initializer(
                stddev=0.01),
            bias_initializer=tf.constant_initializer(0.1),
            activation=tf.nn.elu  # FIXME or perhaps elu?
        )
        print("dense: {}".format(dense.get_shape()))
        dropout = tf.layers.dropout(
            # name="cnn2d_dropout",    #FIXME no name?
            inputs=dense, rate=0.5, training=training)
        return dropout

    @lazy_property
    def cost(self):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.target, logits=self.prediction))
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


class CRnnPredictorV2:

    def __init__(self, data, target, height, training, dropout, num_layers=2, learning_rate=1e-4):
        self.data = data
        self.target = target
        self.training = training
        self.dropout = dropout
        self.training_state = None
        self._height = height
        self._num_layers = num_layers
        self._learning_rate = learning_rate
        self.prediction
        self.accuracy
        self.optimize
        self.logits

    @lazy_property
    def logits(self):
        print("data shape:{}".format(self.data.get_shape()))
        cnn2d = tf.map_fn(lambda x: self.cnn2d(
            x, self.training, self._height), self.data)
        # caught "InvalidArgumentError: Retval[0] does not have value" here
        # cnn2d = tf.layers.batch_normalization(
        #     cnn2d,
        #     beta_initializer=tf.truncated_normal_initializer(0.01),
        #     # gamma_initializer=tf.truncated_normal_initializer(0.01),
        #     moving_mean_initializer=tf.truncated_normal_initializer(0.01),
        #     # moving_variance_initializer=tf.truncated_normal_initializer(0.01),
        #     training=self.training
        # )
        rnn = self.rnn(self, cnn2d)
        rnn = tf.layers.dense(
            inputs=rnn,
            units=rnn.get_shape()[1] * 2,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            bias_initializer=tf.constant_initializer(0.1),
            activation=tf.nn.elu)
        rnn = tf.layers.dropout(
            inputs=rnn, rate=0.5, training=self.training)
        output = tf.layers.dense(
            name="logits",
            inputs=rnn,
            units=int(self.target.get_shape()[1]),
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            bias_initializer=tf.constant_initializer(0.1),
            activation=tf.nn.elu)
        return output

    @staticmethod
    def rnn(self, input):
        # Recurrent network.
        with tf.variable_scope("rnn_parent"):
            cells = []
            state_size = int(input.get_shape()[2]) * 2
            for _ in range(self._num_layers):
                cell = tf.nn.rnn_cell.GRUCell(
                    state_size,
                    kernel_initializer=tf.truncated_normal_initializer(
                        stddev=0.01),
                    bias_initializer=tf.constant_initializer(0.1),
                    activation=tf.nn.elu)  # Or LSTMCell(num_units), or use ConvLSTMCell?
                # cell = tf.nn.rnn_cell.DropoutWrapper(
                #     cell,
                #     output_keep_prob=1.0 - self.dropout)
                cells.append(cell)
            cell = tf.nn.rnn_cell.MultiRNNCell(cells)
            _length = length(input)
            output, self.training_state = tf.nn.dynamic_rnn(
                cell,
                input,
                dtype=tf.float32,
                sequence_length=_length,
                initial_state=self.training_state
            )
            last = last_relevant(output, _length)
            return last

    @staticmethod
    def cnn2d(input, training, height):
        """
        Model function for CNN.
        accepts input shape: [step_size, time_shift*features]
        transformed to: [step_size, time_shift(height), features(width), channel]
        """
        with tf.variable_scope("conv2d_parent"):
            print("shape of cnn input: {}".format(input.get_shape()))
            width = int(input.get_shape()[1])//height
            input2d = tf.reshape(input, [-1, height, width, 1])
            # Transforms into 2D compatible format [batch(step), height, width, channel]
            print(
                "input transformed to 2D pre-conv shape: {}".format(input2d.get_shape()))
            nlayer = numLayers(height, width)
            print("height:{} width:{} #conv layers: {}".format(
                height, width, nlayer))
            filters = max(
                16, 2 ** (math.ceil(math.log(max(height, width), 2))))
            # krange = 3
            # drange = 3
            convlayer = input2d
            for i in range(nlayer):
                filters *= 2
                convlayer = conv2d(convlayer, filters, i)
                # conv = tf.layers.conv2d(
                #     name="conv_lv{}".format(i),
                #     inputs=convlayer,
                #     filters=filters,
                #     kernel_size=2,
                #     kernel_initializer=tf.truncated_normal_initializer(
                #         stddev=0.01),
                #     bias_initializer=tf.constant_initializer(0.1),
                #     padding="same")
                # h_stride = 2 if int(conv.get_shape()[1]) >= 2 else 1
                # w_stride = 2 if int(conv.get_shape()[2]) >= 2 else 1
                # pool = tf.layers.max_pooling2d(
                #     name="pool_lv{}".format(i),
                #     inputs=conv, pool_size=2, strides=[h_stride, w_stride],
                #     padding="same")
                # print("#{} conv:{} pool: {}".format(
                #     i+1, conv.get_shape(), pool.get_shape()))
                # can't use tf.nn.batch_normalization in a mapped function
                # tf.contrib.layers.batch_norm seems ineffective
                # norm = tf.contrib.layers.batch_norm(
                #     inputs=pool,
                #     scale=True,
                #     # updates_collections=None,
                #     # param_initializers=tf.truncated_normal_initializer(
                #     #     stddev=0.01),
                #     is_training=training
                # )
                # norm = tf.nn.lrn(pool, name="lrn_lv{}".format(i))
                # convlayer = pool
            print("final conv2d: {}".format(convlayer.get_shape()))
            convlayer = tf.squeeze(convlayer, [1, 2])
            print("squeezed: {}".format(convlayer.get_shape()))
            # use tf.contrib.layers.fully_connected?
            output = tf.layers.dense(
                name="cnn2d_dense",
                inputs=convlayer,
                units=convlayer.get_shape()[1]*2,
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=0.01),
                bias_initializer=tf.constant_initializer(0.1),
                activation=tf.nn.relu6
            )
            print("dense: {}".format(output.get_shape()))
            output = tf.layers.dropout(
                inputs=output, rate=0.5, training=training)
            return output

    @lazy_property
    def prediction(self):
        with tf.variable_scope("prediction"):
            return tf.argmax(self.logits, 1)

    @lazy_property
    def cost(self):
        with tf.variable_scope("cost"):
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.target, logits=self.logits))
            return cross_entropy

    @lazy_property
    def optimize(self):
        with tf.variable_scope("optimize"):
            update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, "conv2d_parent")
            optimizer = None
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(
                    self.cost, global_step=tf.train.get_global_step())
            return optimizer

    @lazy_property
    def accuracy(self):
        with tf.variable_scope("accuracy"):
            accuracy = tf.equal(tf.argmax(self.target, 1), self.prediction)
            return tf.reduce_mean(tf.cast(accuracy, tf.float32))
