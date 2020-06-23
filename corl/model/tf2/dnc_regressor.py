import tensorflow as tf
import numpy as np

from tensorflow import keras
from time import strftime
from corl.model.tf2.common import DelayedCosineDecayRestarts
from .tf_DNC import dnc

class DNC_Model():
    def __init__(self,
                 output_size=1, controller_units=256, memory_size=256,
                 word_size=64, num_read_heads=4,
                 time_step=30,
                 feat_size=None,
                 dropout_rate=0.5,
                 decayed_dropout_start=None,
                 dropout_decay_steps=None,
                 learning_rate=1e-3,
                 decayed_lr_start=None,
                 lr_decay_steps=None,
                 clipvalue=10
                ):

        self.output_size = output_size
        self.controller_units=controller_units
        self.memory_size=memory_size
        self.word_size=word_size
        self.num_read_heads=num_read_heads

        self._time_step = time_step
        self._feat_size = feat_size
        self._dropout_rate = dropout_rate
        self._decayed_dropout_start = decayed_dropout_start
        self._dropout_decay_steps = dropout_decay_steps
        self._lr = learning_rate
        self._decayed_lr_start = decayed_lr_start
        self._lr_decay_steps = lr_decay_steps
        self._clipvalue = clipvalue

        self.model = None

    def getName(self):
        return self.__class__.__name__

    def getModel(self):
        if self.model is not None:
            return self.model
        print('{} constructing model {}'.format(strftime("%H:%M:%S"),
                                                self.getName()))

        feat = keras.Input(
            shape=(self._time_step, self._feat_size),
            name='features',
            dtype=tf.float32)
        seqlens = keras.Input(shape=(1), name='seqlens', dtype=tf.int32)

        dnc_cell = dnc.DNC(
            self.output_size,
            controller_units=self.controller_units,
            memory_size=self.memory_size,
            word_size=self.word_size,
            num_read_heads=self.num_read_heads
        )
        rnn = keras.layers.RNN(
            dnc_cell, 
            return_sequences=True
        )
        # dnc_initial_state = dnc_cell.get_initial_state(inputs=feat)
        # predictions = rnn(feat, initial_state=dnc_initial_state)
        predictions = rnn(feat)

        inputs = {'features': feat, 'seqlens': seqlens}
        self.model = keras.Model(inputs=inputs, outputs=predictions)
        self.model._name = self.getName()

        return self.model

    def compile(self):
        decay_lr = DelayedCosineDecayRestarts(
            initial_learning_rate=self._lr,
            first_decay_steps=self._lr_decay_steps,
            decay_start=self._decayed_lr_start,
            t_mul=1.02,
            m_mul=0.95,
            alpha=0.095)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=decay_lr,
            # amsgrad=True
            # clipnorm=0.5
            # clipvalue=0.1
            clipvalue=self._clipvalue)
        self.model.compile(
            optimizer=optimizer,
            loss='huber_loss',
            # trying to fix 'Inputs to eager execution function cannot be Keras symbolic tensors'
            # ref: https://github.com/tensorflow/probability/issues/519
            experimental_run_tf_function=False,
            metrics=["mse", "mae"])
        print(self.model.summary())


class DNC_Model_V2(DNC_Model):

    def __init__(self, *args, **kwargs):
        super(DNC_Model_V2, self).__init__(*args, **kwargs)

    def getModel(self):
        if self.model is not None:
            return self.model
        print('{} constructing model {}'.format(strftime("%H:%M:%S"),
                                                self.getName()))

        feat = keras.Input(
            shape=(self._time_step, self._feat_size),
            name='features',
            dtype=tf.float32)
        seqlens = keras.Input(shape=(1), name='seqlens', dtype=tf.int32)

        rnn1 = keras.layers.RNN(
            cell = dnc.DNC(
                name='DNC_1',
                output_size=self.output_size,
                controller_units=self.controller_units,
                memory_size=self.memory_size,
                word_size=self.word_size,
                num_read_heads=self.num_read_heads
            ),
            return_sequences=True
        )
        # dnc_initial_state = dnc_cell.get_initial_state(inputs=feat)
        # predictions = rnn(feat, initial_state=dnc_initial_state)
        rnn1_out = keras.layers.Bidirectional(rnn1)(feat)

        rnn2 = keras.layers.RNN(
            cell = dnc.DNC(
                name='DNC_2',
                output_size=self.output_size,
                controller_units=self.controller_units,
                memory_size=self.memory_size,
                word_size=self.word_size,
                num_read_heads=self.num_read_heads
            ),
            return_sequences=True
        )
        rnn2_out = keras.layers.Bidirectional(rnn2)(rnn1_out)
        rnn2_out = keras.layers.Dropout(self._dropout_rate)(rnn2_out)
        predictions = keras.layers.Dense(1)(rnn2_out)

        inputs = {'features': feat, 'seqlens': seqlens}
        self.model = keras.Model(inputs=inputs, outputs=predictions)
        self.model._name = self.getName()

        return self.model

class DNC_Model_V3(DNC_Model):

    def __init__(self, num_dnc_layers=3, num_fcn_layers=3, *args, **kwargs):
        super(DNC_Model_V3, self).__init__(*args, **kwargs)
        self._num_dnc_layers = num_dnc_layers
        self._num_fcn_layers = num_fcn_layers

    def getModel(self):
        if self.model is not None:
            return self.model
        print('{} constructing model {}'.format(strftime("%H:%M:%S"),
                                                self.getName()))

        feat = keras.Input(
            shape=(self._time_step, self._feat_size),
            name='features',
            dtype=tf.float32)

        # create sequence of DNC layers
        layer = feat
        for i in range(self._num_dnc_layers):
            rnn = keras.layers.RNN(
                cell = dnc.DNC(
                    name='DNC_{}'.format(i),
                    output_size=self.output_size,
                    controller_units=self.controller_units,
                    memory_size=self.memory_size,
                    word_size=self.word_size,
                    num_read_heads=self.num_read_heads
                ),
                return_sequences=True if i+1 < self._num_dnc_layers else False
            )
            layer = keras.layers.Bidirectional(rnn)(layer)
        
        layer = keras.layers.AlphaDropout(self._dropout_rate)(layer)

        # create sequence of FCN layers
        units = self.output_size
        for i in range(self._num_fcn_layer):
            layer = keras.layers.Dense(
                units=units,
                bias_initializer=tf.constant_initializer(0.1),
                activation='selu'
            )(layer)
            units = units // 2

        # Output layer
        outputs = keras.layers.Dense(
            units=1,
            bias_initializer=tf.constant_initializer(0.1),
        )(layer)

        self.model = keras.Model(inputs=feat, outputs=outputs)
        self.model._name = self.getName()

        return self.model