from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow import keras
from time import strftime


class GlobalStepMarker(keras.layers.Layer):
    '''
        Record global steps.
    '''
    def __init__(self, time_step=None, feat_size=None):
        super(GlobalStepMarker, self).__init__()
        self.time_step = time_step
        self.feat_size = feat_size

        # self.seqlen = None
        # self.feat = None

        # self.feat = keras.Input(
        #     # A shape tuple (integers), not including the batch size.
        #     shape=(self.time_step, self.feat_size),
        #     # The name becomes a key in the dict.
        #     name='features',
        #     dtype='float32')
        # self.seqlens = keras.Input(
        #     shape=(1),
        #     # The name becomes a key in the dict.
        #     name='seqlens',
        #     dtype='int32')
        # self.inputs = {'features': self.feat, 'seqlens': self.seqlens}

    def build(self, input_shape):
        super(GlobalStepMarker, self).build(input_shape)
        self.global_step = tf.Variable(initial_value=0,
                                       trainable=False,
                                       name="global_step")
        # self.seqlen = tf.Variable(name="seqlen",
        #                           initial_value=tf.zeros((1, 1), tf.int32),
        #                           dtype=tf.int32,
        #                           validate_shape=False,
        #                           trainable=False,
        #                           shape=(None, 1)
        #                           #   shape=tf.TensorShape([None, 1])
        #                           )

        # self.feat = tf.Variable(
        #     name="features",
        #     initial_value=tf.zeros((1, self.time_step, self.feat_size),
        #                            tf.float32),
        #     trainable=False,
        #     dtype=tf.float32,
        #     shape=tf.TensorShape([None, self.time_step, self.feat_size]))

    def getGlobalStep(self):
        return self.global_step

    def call(self, inputs):
        self.global_step.assign_add(1)
        # feat = inputs['features']
        # seqlens = inputs['seqlens']
        # self.feat.assign(feat)
        # self.seqlen.assign(seqlens)
        return inputs

    def get_config(self):
        config = super().get_config().copy()
        # config.update({
        #     'num_layers': self.num_layers,
        # })
        return config


class LastRelevant(keras.layers.Layer):
    def __init__(self):
        super(LastRelevant, self).__init__()

    def call(self, inputs):
        lstm, seqlens = inputs
        batch_size = tf.shape(lstm)[0]
        out = tf.gather_nd(
            lstm,
            tf.stack([tf.range(batch_size),
                      tf.reshape(seqlens, [-1]) - 1],
                     axis=1))
        # out = tf.reshape(out, shape=(-1, 1))
        return out

    def get_config(self):
        config = super().get_config().copy()
        # config.update({
        #     'num_layers': self.num_layers,
        # })
        return config


class Squeeze(keras.layers.Layer):
    def __init__(self):
        super(Squeeze, self).__init__()

    def call(self, inputs):
        output = tf.squeeze(inputs)
        output.set_shape([None])
        return output

    def get_config(self):
        config = super().get_config().copy()
        # config.update({
        #     'num_layers': self.num_layers,
        # })
        return config


class DropoutRate:
    def __init__(self, rate, decayed_dropout_start, dropout_decay_steps, seed):
        self._rate = rate
        self._decayed_dropout_start = decayed_dropout_start
        self._dropout_decay_steps = dropout_decay_steps
        self._seed = seed

    def call(self):
        gstep = tf.compat.v1.train.get_or_create_global_step()

        def kp():
            return tf.multiply(self._kp, 1.0)

        def cdr_kp():
            return 1.0 - tf.compat.v1.train.cosine_decay_restarts(
                learning_rate=self._rate,
                global_step=gstep - self._decayed_dropout_start,
                first_decay_steps=self._dropout_decay_steps,
                t_mul=1.05,
                m_mul=0.98,
                alpha=0.01)

        minv = kp()
        if self._decayed_dropout_start is not None:
            minv = tf.cond(pred=tf.less(gstep, self._decayed_dropout_start),
                           true_fn=kp,
                           false_fn=cdr_kp)
        rdu = tf.random.uniform([],
                                minval=minv,
                                maxval=1.02,
                                dtype=tf.float32,
                                seed=self._seed)
        return tf.minimum(1.0, rdu)


class Infer(keras.layers.Layer):
    '''
    Returns positive code, positive corl, negative code, negative corl
    '''
    def __init__(self):
        super(Infer, self).__init__()

    def call(self, inputs):
        pos_idx = tf.argmax(input=inputs)
        neg_idx = tf.argmin(input=inputs)
        posc = tf.gather(self.refs, pos_idx)
        pcorl = tf.gather(inputs, pos_idx)
        negc = tf.gather(self.refs, neg_idx)
        ncorl = tf.gather(inputs, neg_idx)
        return posc, pcorl, negc, ncorl


class Worst(keras.layers.Layer):
    def __init__(self):
        super(Worst, self).__init__()

    def call(self, inputs):
        sqd = tf.math.squared_difference(inputs, self.target)
        bidx = tf.argmax(input=sqd)
        max_diff = tf.sqrt(tf.reduce_max(input_tensor=sqd))
        predict = tf.gather(inputs, bidx)
        actual = tf.gather(self.target, bidx)
        return max_diff, predict, actual


@tf.function
def getInputs(time_step, feat_size):
    feat = keras.Input(
        # A shape tuple (integers), not including the batch size.
        shape=(time_step, feat_size),
        # The name becomes a key in the dict.
        name='features',
        dtype='float32')
    seqlens = keras.Input(
        shape=(),
        # The name becomes a key in the dict.
        name='seqlens',
        dtype='int32')
    return [feat, seqlens]


class LSTMRegressorV1:
    '''

    '''
    def __init__(self,
                 layer_width=200,
                 time_step=30,
                 feat_size=3,
                 dropout_rate=0.5,
                 decayed_dropout_start=None,
                 dropout_decay_steps=None,
                 learning_rate=1e-3,
                 decayed_lr_start=None,
                 lr_decay_steps=None,
                 seed=None):
        self._layer_width = layer_width
        self._time_step = time_step
        self._feat_size = feat_size
        self._dropout_rate = dropout_rate
        self._decayed_dropout_start = decayed_dropout_start
        self._dropout_decay_steps = dropout_decay_steps
        self._lr = learning_rate
        self._decayed_lr_start = decayed_lr_start
        self._lr_decay_steps = lr_decay_steps
        self._seed = seed
        self.model = None

    def getName(self):
        return self.__class__.__name__

    def getModel(self):
        if self.model is not None:
            return self.model
        print('{} constructing model {}'.format(strftime("%H:%M:%S"),
                                                self.getName()))
        feat = keras.Input(
            # A shape tuple (integers), not including the batch size.
            shape=(self._time_step, self._feat_size),
            # The name becomes a key in the dict.
            name='features',
            dtype='float32')
        seqlens = keras.Input(
            shape=(1),
            # The name becomes a key in the dict.
            name='seqlens',
            dtype='int32')
        inputs = {'features': feat, 'seqlens': seqlens}

        # feat = tf.debugging.check_numerics(feat, 'NaN/Inf found in feat')
        # seqlens = tf.debugging.check_numerics(seqlens, 'NaN/Inf found in seqlens')

        # inputs = getInputs(self._time_step, self._feat_size)
        # seqlens = inputs[1]
        # gsm = GlobalStepMarker()(feat)
        # inputs = inputLayer.getInputs()

        # RNN
        # choice for regularizer:
        # https://machinelearningmastery.com/use-weight-regularization-lstm-networks-time-series-forecasting/
        reg = keras.regularizers.l1_l2(0.01, 0.01)
        lstm = keras.layers.LSTM(
            units=self._layer_width,
            # stateful=True,
            return_sequences=True,
            kernel_initializer=keras.initializers.VarianceScaling(),
            bias_initializer=tf.constant_initializer(0.1),
            kernel_regularizer=reg,
            # recurrent_regularizer=reg,
        )(feat)
        lstm = keras.layers.LSTM(
            units=self._layer_width // 2,
            # stateful=True,
            return_sequences=True,
            kernel_initializer=keras.initializers.VarianceScaling(),
            bias_initializer=tf.constant_initializer(0.1),
            kernel_regularizer=reg,
            # recurrent_regularizer=reg,
        )(lstm)
        # extract last_relevant timestep
        lstm = LastRelevant()((lstm, seqlens))

        # FCN
        fcn = keras.layers.Dense(
            units=self._layer_width // 2,
            #  kernel_initializer='lecun_normal',
            kernel_initializer='variance_scaling',
            bias_initializer=tf.constant_initializer(0.1),
            activation='selu')(lstm)
        fsize = self._layer_width // 2
        nlayer = 3
        for i in range(nlayer):
            if i == 0:
                fcn = keras.layers.AlphaDropout(rate=self._dropout_rate)(fcn)
            fcn = keras.layers.Dense(
                units=fsize,
                #  kernel_initializer='lecun_normal',
                kernel_initializer='variance_scaling',
                bias_initializer=tf.constant_initializer(0.1),
            )(fcn)
            fsize = fsize // 2
        fcn = keras.layers.Dense(
            units=fsize,
            #  kernel_initializer='lecun_normal',
            kernel_initializer='variance_scaling',
            bias_initializer=tf.constant_initializer(0.1),
            activation='selu')(fcn)

        # Output layer
        outputs = keras.layers.Dense(
            units=1,
            # kernel_initializer='lecun_normal',
            kernel_initializer='variance_scaling',
            bias_initializer=tf.constant_initializer(0.1),
        )(fcn)
        # outputs = tf.squeeze(outputs)

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model._name = self.getName()

        return self.model

    def compile(self):
        # TODO study how to use ReduceLROnPlateau and CosineDecayRestarts on adam optimizer
        # decay = tf.keras.experimental.CosineDecayRestarts(self._lr,
        #                                                   self._lr_decay_steps,
        #                                                   t_mul=1.02,
        #                                                   m_mul=0.95,
        #                                                   alpha=0.095)

        # optimizer = tf.keras.optimizers.Adam(
        #     learning_rate=self._lr,
        #     # amsgrad=True
        #     # clipnorm=32
        #     clipvalue=0.15)

        optimizer = keras.optimizers.Nadam(learning_rate=self._lr,
                                           clipvalue=0.15)

        self.model.compile(
            optimizer=optimizer,
            loss='mae',
            # metrics=[
            #     # Already in the "loss" metric
            #     'mse',
            #     # Only available for logistic regressor with prediction >= 0
            #     'accuracy'
            #     # keras.metrics.Precision(),
            #     # keras.metrics.Recall()
            # ],

            # trying to fix 'Inputs to eager execution function cannot be Keras symbolic tensors'
            # ref: https://github.com/tensorflow/probability/issues/519
            experimental_run_tf_function=False)

        #TypeError: Error converting shape to a TensorShape: Dimension value must be integer or None or have an __index__ method, got [35, 6].
        #input_shape = ([self._time_step, self._feat_size], None)
        # input_shape = {[self._time_step, self._feat_size], None}
        # self.model.build(input_shape)
        print(self.model.summary())
