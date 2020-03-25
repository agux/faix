from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow import keras


class InputLayer(keras.layers.Layer):
    '''
        Serve as entry point to split features and seqlens from the dictionary in x.
    '''
    def __init__(self):
        super(InputLayer, self).__init__()
        self.seqlens = tf.Variable(trainable=False,
                                   validate_shape=True,
                                   name='seqlens',
                                   dtype=tf.int32,
                                   shape=[None])

    def getSeqLens(self):
        return self.seqlens

    def call(self, inputs):
        self.seqlens.assign(value=inputs['seqlens'], name='update_seqlens')
        return inputs['features']


class LastRelevant(keras.layers.Layer):
    def __init__(self, seqlens):
        super(LastRelevant, self).__init__()
        self.seqlens = seqlens

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        return tf.gather_nd(
            inputs, tf.stack([tf.range(batch_size), self.seqlens - 1], axis=1))


class Squeeze(keras.layers.Layer):
    def __init__(self):
        super(Squeeze, self).__init__()

    def call(self, inputs):
        output = tf.squeeze(inputs)
        output.set_shape([None])
        return output


class DropoutRate:
    def __init__(self, rate, decayed_dropout_start, dropout_decay_steps, seed):
        self._rate = rate
        self._decayed_dropout_start
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


class LSTMRegressorV1:
    '''

    '''
    def __init__(self,
                 layer_width=200,
                 dropout_rate=0.5,
                 decayed_dropout_start=None,
                 dropout_decay_steps=None,
                 learning_rate=1e-3,
                 decayed_lr_start=None,
                 lr_decay_steps=None,
                 seed=None):
        self._layer_width = layer_width
        self._dropout_rate = dropout_rate
        self._decayed_dropout_start = decayed_dropout_start
        self._dropout_decay_steps = dropout_decay_steps
        self._lr = learning_rate
        self._decayed_lr_start = decayed_lr_start
        self._lr_decay_steps = lr_decay_steps
        self._seed = seed

    def getName(self):
        return self.__class__.__name__

    def setModel(self, model):
        self.model = model

    @staticmethod
    def getModel(self):
        if self.model is not None:
            return self.model

        inputLayer = InputLayer()
        self.model = keras.Sequential()
        self.model.add(inputLayer)
        # RNN
        self.model.add(
            keras.layers.LSTM(units=self._layer_width,
                              return_sequences=True,
                              input_shape=self.data.shape[-2:]))
        self.model.add(keras.layers.LSTM(units=self._layer_width // 2))

        # extract last_relevant timestep
        self.model.add(LastRelevant(inputLayer.getSeqLens()))

        # Activation
        self.model.add(
            keras.layers.Dense(units=self._layer_width // 2,
                               kernel_initializer='lecun_normal',
                               activation='selu'))

        # FCN
        fsize = self._layer_width // 2
        nlayer = 3
        for i in range(nlayer):
            if i == 0:
                self.model.add(
                    keras.layers.AlphaDropout(rate=DropoutRate(
                        self._dropout_rate, self._decayed_dropout_start,
                        self._dropout_decay_steps, self._seed)))
            self.model.add(
                keras.layers.Dense(units=fsize,
                                   kernel_initializer='lecun_normal'))
            fsize = fsize // 2
        self.model.add(
            keras.layers.Dense(units=fsize,
                               kernel_initializer='lecun_normal',
                               activation='selu'))

        # Output layer
        self.model.add(
            keras.layers.Dense(
                units=1,
                kernel_initializer='lecun_normal',
            ))
        self.model.add(Squeeze())

        # TODO study how to use ReduceLROnPlateau and CosineDecayRestarts on adam optimizer
        decay = tf.keras.experimental.CosineDecayRestarts(self._lr,
                                                          self._lr_decay_steps,
                                                          t_mul=1.02,
                                                          m_mul=0.95,
                                                          alpha=0.095)
        adam = tf.keras.optimizers.Adam(learning_rate=decay)

        self.model.compile(optimizer=adam,
                           loss='mse',
                           metrics=['accuracy', 'loss', 'precision', 'recall'])

        self.model.summary()
        return self.model
