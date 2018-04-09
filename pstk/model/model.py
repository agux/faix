from __future__ import print_function
import functools
import tensorflow as tf
import numpy as np
import math


def weight_bias(W_shape, b_shape, bias_init=0.1):
    W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1), name='weight')
    b = tf.Variable(tf.constant(bias_init, shape=b_shape), name='bias')
    return W, b


def highway(x, activation=None, carry_bias=-1.0):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).

    t = sigmoid(W_T*x + b_T)
    y = t * g(Wx + b) + (1 - t) * x
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.

    the weight(W_T,W) in highway layer must have same size,but you can use fully-connected layers to change dimensionality. 

    with larger negative carry_bias, more input (x) will be kept in the final output of highway layer.

    """
    with tf.name_scope("highway"):
        size = int(x.get_shape()[1])
        W, b = weight_bias([size, size], [size])

        with tf.name_scope('transform_gate'):
            W_T, b_T = weight_bias([size, size], [size], bias_init=carry_bias)

        H = tf.matmul(x, W) + b
        if activation is not None:
            H = activation(H, name='activation')
        T = tf.sigmoid(tf.matmul(x, W_T) + b_T, name='transform_gate')
        C = tf.subtract(1.0, T, name="carry_gate")

        # y = (H * T) + (x * C)
        y = tf.add(tf.multiply(H, T), tf.multiply(x, C), name='y')
        return y


def dense_block(input, width):
    output = tf.layers.dense(
        inputs=input,
        units=width,
        kernel_initializer=tf.truncated_normal_initializer(
            stddev=0.01),
        bias_initializer=tf.constant_initializer(0.1)
    )
    output = tf.concat([input, output], -1)
    return output


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


def numLayers(d1, d2=None):
    n1 = 0
    while d1 > 1:
        d1 = math.ceil(d1/2.0)
        n1 += 1
    n2 = 0
    if d2 is not None:
        n2 = numLayers(d2)
    return max(n1, n2)


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


class SecurityGradePredictor:

    def __init__(self, data, target, wsize, training, is3d=False, num_hidden=200, num_layers=2, learning_rate=1e-4):
        self.data = data
        self.target = target
        self.training = training
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self._learning_rate = learning_rate
        self._is3d = is3d
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
        filters = max(2, 2 ** (math.ceil(math.log(feat, 2)+1)))
        krange = 3
        drange = 3
        convlayers = np.array([[input for _ in range(drange)]
                               for _ in range(krange)])
        for i in range(nlayer):
            filters *= 2
            uf = math.ceil(filters/(krange*drange))
            for k in range(krange):
                for d in range(drange):
                    conv = tf.layers.separable_conv2d(
                        inputs=convlayers[k][d],
                        filters=uf,
                        depth_multiplier=3,
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
        output_layer = tf.layers.batch_normalization(
            output_layer, training=self.training)
        print("cnn output layer: {}".format(output_layer.get_shape()))
        # self.data = output_layer
        return output_layer

    @lazy_property
    def multi_cnn3d(self):
        """Model function for 3D CNN."""
        step = int(self.data.get_shape()[1])
        feat = int(self.data.get_shape()[2])
        # Get 2D dimension length (height, width)
        h, w = factorize(feat)
        # Transforms input to 3D [batch, depth, height, width, channel]
        input = tf.reshape(self.data, [-1, step, h, w, 1])
        print("input transformed to 3D shape: {}".format(input.get_shape()))
        nlayer = numLayers(h, w)
        wsize = self._wsize
        print("window size:{} step:{} #conv layers: {}".format(
            wsize, step, nlayer))
        filters = max(2, 2 ** (math.ceil(math.log(feat, 2)+1)))
        # krange = min(h, w) // 2
        # drange = krange
        krange = 1
        drange = 1
        convlayers = np.array([[input for _ in range(drange)]
                               for _ in range(krange)])
        for i in range(nlayer):
            filters *= 4
            uf = math.ceil(filters/(krange*drange))
            for k in range(krange):
                for d in range(drange):
                    conv = tf.layers.conv3d(
                        inputs=convlayers[k][d],
                        filters=uf,
                        kernel_size=(k+wsize, k+2, k+2),
                        kernel_initializer=tf.truncated_normal_initializer(
                            stddev=0.01),
                        bias_initializer=tf.constant_initializer(0.1),
                        dilation_rate=(max(2, d+1), d+1, d+1),
                        padding="same",
                        activation=tf.nn.elu)
                    h_stride = 2 if int(conv.get_shape()[2]) >= 2 else 1
                    w_stride = 2 if int(conv.get_shape()[3]) >= 2 else 1
                    pool = tf.layers.max_pooling3d(
                        inputs=conv, pool_size=k+2, strides=[1, h_stride, w_stride],
                        padding="same")
                    convlayers[k][d] = pool
                    print("#{} conv:{} pool: {} wide: {} dilation: {}".format(
                        i+1, conv.get_shape(), pool.get_shape(), k+2, d+1))
        # Flatten convlayers
        convlayers = convlayers.flatten()
        convlayer = tf.concat([c for c in convlayers], 4)
        print("concat: {}".format(convlayer.get_shape()))
        output_layer = tf.squeeze(convlayer, [2, 3])
        output_layer = tf.layers.batch_normalization(
            output_layer, training=self.training)
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
        return self.crnn2d
        # Recurrent network.
        # cells = []
        # for _ in range(self._num_layers):
        #     cell = tf.nn.rnn_cell.GRUCell(
        #         self._num_hidden)  # Or LSTMCell(num_units), or use ConvLSTMCell?
        #     # cell = tf.nn.rnn_cell.DropoutWrapper(
        #     #     cell, output_keep_prob=1.0 - self.dropout)
        #     cells.append(cell)
        # cell = tf.nn.rnn_cell.MultiRNNCell(cells)

        # output, _ = tf.nn.dynamic_rnn(
        #     cell,
        #     self.multi_cnn3d if self._is3d else self.multi_cnn,
        #     dtype=tf.float32,
        #     sequence_length=self.length,
        # )

        # last = self._last_relevant(output, self.length)
        # weight, bias = self._weight_and_bias(
        #     self._num_hidden, int(self.target.get_shape()[1]))
        # prediction = tf.matmul(last, weight) + bias

        # norm_last = tf.layers.batch_normalization(
        #     last, training=self.training)
        # dense = tf.layers.dense(
        #     inputs=last, units=self._num_hidden * 3, activation=tf.nn.elu)
        # dropout = tf.layers.dropout(
        #     inputs=dense, rate=math.e/10, training=self.training)

        # Logits Layer
        # prediction = tf.layers.dense(
        #     inputs=dense, units=int(self.target.get_shape()[1]), activation=tf.nn.relu6)

        # prediction = self._cnn_layer(self.data,
        #                              self._ksize,
        #                              int(self.target.get_shape()[1]),
        #                              self.training)
        # return prediction

    @lazy_property
    def crnn2d(self):
        # map [batch, step, feature] to [batch][step, feature] and pass each
        # [step, feature] to cnn2d
        cnn = tf.map_fn(lambda input: self.cnn2d(
            input, self.training), self.data)
        # can't use batch_norm due to its bug causing "InvalidArgumentError: Retval[0] does not have value"
        # norm_cnn = tf.layers.batch_normalization(cnn, training=self.training)
        mix = tf.concat([self.data, cnn], 2)
        rnn = self.rnn(self, mix)

        dense = tf.layers.dense(
            inputs=rnn,
            units=self._num_hidden * 3,
            # kernel_initializer=tf.truncated_normal_initializer,
            # bias_initializer=tf.constant_initializer(0.1),
            activation=tf.nn.elu)
        dropout = tf.layers.dropout(
            inputs=dense, rate=math.e/10, training=self.training)

        # Logits Layer
        output = tf.layers.dense(
            inputs=dropout,
            units=int(self.target.get_shape()[1]),
            # kernel_initializer=tf.truncated_normal_initializer,
            # bias_initializer=tf.constant_initializer(0.1),
            activation=tf.nn.relu6)
        return output

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
        # optimizer = tf.train.AdamOptimizer(self._learning_rate)
        # return optimizer.minimize(self.cost, global_step=tf.train.get_global_step())

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
    def cnn2d(input, training):
        """Model function for CNN."""
        print("shape of cnn input: {}".format(input.get_shape()))
        step = int(input.get_shape()[0])
        feat = int(input.get_shape()[1])
        # Get 2D dimension length (height, width)
        h, w = factorize(feat)
        # Transforms into 2D compatible format [batch(step), height, width, channel]
        input2d = tf.reshape(input, [-1, h, w, 1])
        print("input transformed to 2D shape: {}".format(input2d.get_shape()))

        nlayer = numLayers(h, w)
        print("step:{} feat:{} #conv layers: {}".format(step, feat, nlayer))
        filters = max(2, 2 ** (math.ceil(math.log(feat, 2)+1)))
        # krange = 3
        # drange = 3
        krange = 1
        drange = 1
        convlayers = np.array([[input2d for _ in range(drange)]
                               for _ in range(krange)])
        for i in range(nlayer):
            filters *= 2
            uf = math.ceil(filters/(krange*drange))
            for k in range(krange):
                for d in range(drange):
                    conv = tf.layers.conv2d(
                        inputs=convlayers[k][d],
                        filters=uf,
                        kernel_size=k+2,
                        kernel_initializer=tf.truncated_normal_initializer(
                            stddev=0.01),
                        bias_initializer=tf.constant_initializer(0.1),
                        dilation_rate=d+1,
                        padding="same",
                        activation=tf.nn.elu)
                    h_stride = 2 if int(conv.get_shape()[1]) >= 2 else 1
                    w_stride = 2 if int(conv.get_shape()[2]) >= 2 else 1
                    pool = tf.layers.max_pooling2d(
                        inputs=conv, pool_size=k+2, strides=[h_stride, w_stride],
                        padding="same")
                    # can't use for now due to map_fn and batch_norm cooperation bugs
                    # norm_pool = tf.layers.batch_normalization(
                    #     pool, training=training)
                    convlayers[k][d] = pool
                    print("#{} conv:{} pool: {} ksize: {} dilation: {}".format(
                        i+1, conv.get_shape(), pool.get_shape(), k+2, d+1))
        # Flatten convlayers
        convlayers = convlayers.flatten()
        convlayer = tf.concat([c for c in convlayers], 3)
        print("concat: {}".format(convlayer.get_shape()))
        convlayer = tf.squeeze(convlayer, [1, 2])
        print("squeeze: {}".format(convlayer.get_shape()))
        units = 2 ** math.ceil(math.log(feat*3, 2))-feat
        dense = tf.layers.dense(
            inputs=convlayer,
            units=units,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            bias_initializer=tf.constant_initializer(0.1),
            activation=tf.nn.elu)
        # can't use for now due to map_fn and batch_norm cooperation bugs
        # norm_dense = tf.layers.batch_normalization(
        #     dense, training=training)
        # output_layer = tf.concat([input, norm_dense], 1)
        print("cnn output layer: {}".format(dense.get_shape()))
        return dense

    @staticmethod
    def rnn(self, input):
        print("rnn input: {}".format(input.get_shape()))
        # Recurrent network.
        cells = []
        for _ in range(self._num_layers):
            cell = tf.nn.rnn_cell.GRUCell(
                self._num_hidden,
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=0.01),
                bias_initializer=tf.constant_initializer(0.1))  # Or LSTMCell(num_units), or use ConvLSTMCell?
            # cell = tf.nn.rnn_cell.DropoutWrapper(
            #     cell, output_keep_prob=1.0 - self.dropout)
            cells.append(cell)
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)

        output, _ = tf.nn.dynamic_rnn(
            cell,
            input,
            dtype=tf.float32,
            sequence_length=self.length,
        )

        last = last_relevant(output, self.length)
        # weight, bias = self._weight_and_bias(
        #     self._num_hidden, int(self.target.get_shape()[1]))
        # prediction = tf.matmul(last, weight) + bias

        # norm_last = tf.layers.batch_normalization(
        #     last, training=self.training)
        return last

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


def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = int(output.get_shape()[1])
    output_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, output_size])
    relevant = tf.gather(flat, index)
    return relevant


class BasicRnnPredictor:

    def __init__(self, data, target, training, dropout, num_hidden=200, num_layers=2, learning_rate=1e-4):
        self.data = data
        self.target = target
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
    def length(self):
        used = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    @lazy_property
    def prediction(self):
        rnn = self.rnn(self, self.data)
        dense = tf.layers.dense(
            inputs=rnn,
            units=self._num_hidden * 3,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            bias_initializer=tf.constant_initializer(0.1),
            activation=tf.nn.elu)
        dropout = tf.layers.dropout(
            inputs=dense, rate=0.5, training=self.training)
        output = tf.layers.dense(
            inputs=dropout,
            units=int(self.target.get_shape()[1]),
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            bias_initializer=tf.constant_initializer(0.1),
            activation=tf.nn.relu6)
        return output

    @staticmethod
    def rnn(self, input):
        # Recurrent network.
        cells = []
        for _ in range(self._num_layers):
            cell = tf.nn.rnn_cell.GRUCell(
                self._num_hidden,
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=0.01),
                bias_initializer=tf.constant_initializer(0.1))  # Or LSTMCell(num_units), or use ConvLSTMCell?
            # cell = tf.nn.rnn_cell.DropoutWrapper(
            #     cell, output_keep_prob=1.0 - self.dropout)
            cells.append(cell)
            # cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
            #     self._num_hidden,
            #     dropout_keep_prob=1.0 - self.dropout
            # )
            # cells.append(cell)
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)

        output, self.training_state = tf.nn.dynamic_rnn(
            cell,
            input,
            dtype=tf.float32,
            sequence_length=self.length,
            initial_state=self.training_state
        )

        last = last_relevant(output, self.length)
        return last

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


class BasicCnnPredictor:

    def __init__(self, data, target, wsize, training, span=5, num_hidden=200, num_layers=2, learning_rate=1e-4):
        self.data = data
        self.target = target
        self.training = training
        self.training_state = None
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self._learning_rate = learning_rate
        self._span = span
        self._wsize = wsize
        self.prediction
        self.accuracy
        self.optimize

    @lazy_property
    def length(self):
        used = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    @lazy_property
    def prediction(self):
        cnn = self.cnn2d(self.data, self.training)
        output = tf.layers.dense(
            inputs=cnn,
            units=int(self.target.get_shape()[1]),
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            bias_initializer=tf.constant_initializer(0.1),
            activation=tf.nn.relu6)
        return output

    @staticmethod
    def cnn2d(input, training):
        """Model function for CNN."""
        print("shape of cnn input: {}".format(input.get_shape()))
        step = int(input.get_shape()[1])
        feat = int(input.get_shape()[2])
        # Transforms into 2D compatible format [batch(step), height, width, channel]
        input2d = tf.reshape(input, [-1, step, feat, 1])
        print("input transformed to 2D shape: {}".format(input2d.get_shape()))

        nlayer = numLayers(step, feat)
        print("step:{} feat:{} #conv layers: {}".format(step, feat, nlayer))
        filters = max(2, 2 ** (math.ceil(math.log(feat, 2))))
        # krange = 3
        # drange = 3
        krange = 1
        drange = 1
        convlayers = np.array([[input2d for _ in range(drange)]
                               for _ in range(krange)])
        for i in range(nlayer):
            filters *= 2
            uf = math.ceil(filters/(krange*drange))
            for k in range(krange):
                for d in range(drange):
                    conv = tf.layers.conv2d(
                        inputs=convlayers[k][d],
                        filters=uf,
                        kernel_size=k+2,
                        kernel_initializer=tf.truncated_normal_initializer(
                            stddev=0.01),
                        bias_initializer=tf.constant_initializer(0.1),
                        dilation_rate=d+1,
                        padding="same",
                        activation=tf.nn.elu)
                    h_stride = 2 if int(conv.get_shape()[1]) >= 2 else 1
                    w_stride = 2 if int(conv.get_shape()[2]) >= 2 else 1
                    pool = tf.layers.max_pooling2d(
                        inputs=conv, pool_size=k+2, strides=[h_stride, w_stride],
                        padding="same")
                    norm_pool = tf.layers.batch_normalization(
                        pool, training=training)
                    convlayers[k][d] = norm_pool
                    print("#{} conv:{} pool: {} ksize: {} dilation: {}".format(
                        i+1, conv.get_shape(), pool.get_shape(), k+2, d+1))
        # Flatten convlayers
        convlayers = convlayers.flatten()
        convlayer = tf.concat([c for c in convlayers], 3)
        print("concat: {}".format(convlayer.get_shape()))
        convlayer = tf.squeeze(convlayer, [1, 2])
        print("squeeze: {}".format(convlayer.get_shape()))
        # dropout = tf.layers.dropout(
        #     inputs=convlayer, rate=0.5, training=training)
        return convlayer

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


def length(data):
    used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


class ShiftRnnPredictor:

    def __init__(self, data, target, training, dropout, num_hidden=200, num_layers=2, learning_rate=1e-4):
        self.data = data
        self.target = target
        self.training = training
        self.dropout = dropout
        self._shifts = int(data.shape[0])
        self.training_state = [None for _ in range(self._shifts+1)]
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self._learning_rate = learning_rate
        self.prediction
        self.accuracy
        self.optimize

    @lazy_property
    def prediction(self):
        mrnn = []
        slots = ["c"+str(i) for i in range(self._shifts)]
        print("data: {}".format(self.data.get_shape()))
        mrnn = tf.map_fn(lambda x:
                         self.rnn(x, self._num_layers, self._num_hidden, self.dropout, slots), self.data)
        print("mrnn: {}".format(mrnn.get_shape()))
        mrnn = tf.stack(mrnn)
        mrnn = tf.transpose(mrnn, [1, 0, 2])
        print("stacked & transposed: {}".format(mrnn.get_shape()))
        rnn = self.rnn(mrnn, self._num_layers, self._num_hidden,
                       self.dropout, ["parent"])
        print("rnn: {}".format(rnn.get_shape()))
        # dense = tf.layers.dense(
        #     inputs=rnn,
        #     units=self._num_hidden * 3,
        #     kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        #     bias_initializer=tf.constant_initializer(0.1),
        #     activation=tf.nn.elu)
        # dropout = tf.layers.dropout(
        #     inputs=rnn, rate=0.5, training=self.training)
        output = tf.layers.dense(
            inputs=rnn,
            units=int(self.target.get_shape()[1]),
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            bias_initializer=tf.constant_initializer(0.1),
            activation=tf.nn.relu6)
        return output

    @staticmethod
    def rnn(data, num_layers, num_hidden, dropout, slots):
        # Recurrent network.
        with tf.variable_scope("rnn_{}".format(slots.pop())):
            cells = []
            for _ in range(num_layers):
                cell = tf.nn.rnn_cell.GRUCell(
                    num_hidden,
                    kernel_initializer=tf.truncated_normal_initializer(
                        stddev=0.01),
                    bias_initializer=tf.constant_initializer(0.1))  # Or LSTMCell(num_units), or use ConvLSTMCell?
                cell = tf.nn.rnn_cell.DropoutWrapper(
                    cell, output_keep_prob=1.0 - dropout)
                # cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                #     num_hidden,
                #     dropout_keep_prob=1.0 - dropout
                # )
                cells.append(cell)
            cell = tf.nn.rnn_cell.MultiRNNCell(cells)
            _length = length(data)
            output, _ = tf.nn.dynamic_rnn(
                cell,
                data,
                dtype=tf.float32,
                sequence_length=_length
            )
            last = last_relevant(output, _length)
            dense = tf.layers.dense(
                inputs=last,
                units=last.get_shape()[1] * 2,
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=0.01),
                bias_initializer=tf.constant_initializer(0.1),
                activation=tf.nn.relu6)
            # norm = tf.layers.batch_normalization(
            #     dense, training=self.training)
            return dense

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
