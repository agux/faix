import tensorflow as tf
from tensorflow import keras
from time import strftime

from .addon.tf_DNC import dnc
from .common import DelayedCosineDecayRestarts, CausalConv1D_V2, DecayedDropoutLayer

# TODO rafactor this model to support stock trend classification.


class BaseModel():
    '''

        Multiple causal Conv1D layers on same input, BatchNorm, Multiple Bidirectional DNC layers, (DecayedDropout#1), Multiple FCN layers (DecayedDropout#2)

        * DecayedDropout#1 exists only if number of FCN layer is 0
        * DecayedDropout#2 exists only if number of FCN layer is > 0

    '''

    def __init__(self,
                 output_size=1,
                 controller_units=256,
                 memory_size=256,
                 word_size=64,
                 num_read_heads=4,
                 time_step=30,
                 feat_size=None,
                 dropout_rate=0.5,
                 decayed_dropout_start=None,
                 dropout_decay_steps=None,
                 learning_rate=1e-3,
                 decayed_lr_start=None,
                 lr_decay_steps=None,
                 clipvalue=10,
                 num_cnn_layers=1,
                 num_dnc_layers=2,
                 num_fcn_layers=2,
                 cnn_filters=64,  # can be a list
                 cnn_kernel_size=3,  # can be a list
                 cnn_output_size=256,
                 layer_norm_lstm=False,
                 num_classes=5,
                 seed=None):
        self.output_size = output_size
        self.controller_units = controller_units
        self.memory_size = memory_size
        self.word_size = word_size
        self.num_read_heads = num_read_heads

        self._time_step = time_step
        self._feat_size = feat_size
        self._dropout_rate = dropout_rate
        self._decayed_dropout_start = decayed_dropout_start
        self._dropout_decay_steps = dropout_decay_steps
        self._lr = learning_rate
        self._decayed_lr_start = decayed_lr_start
        self._lr_decay_steps = lr_decay_steps
        self._clipvalue = clipvalue

        self._num_cnn_layers = num_cnn_layers
        self._num_dnc_layers = num_dnc_layers
        self._num_fcn_layers = num_fcn_layers
        self._cnn_filters = cnn_filters
        self._cnn_kernel_size = cnn_kernel_size
        self._cnn_output_size = cnn_output_size
        self._layer_norm_lstm = layer_norm_lstm

        self._num_classes = num_classes

        self.seed = seed

        self.model = None

    def getName(self):
        return self.__class__.__name__

    def _inputLayer(self):
        return keras.Input(
            shape=(self._time_step, self._feat_size),
            dtype=tf.float32
        )

    def getModel(self):
        if self.model is not None:
            return self.model
        print('{} constructing model {}'.format(strftime("%H:%M:%S"),
                                                self.getName()))

        inputs = self._inputLayer()

        layer = inputs
        # CNN Layers
        if self._num_cnn_layers > 0:
            layer = CausalConv1D_V2(
                self._num_cnn_layers,
                self._cnn_filters,
                self._cnn_kernel_size,
                'selu',
                name='CausalCNN'
            )(layer)
            layer = keras.layers.Dense(
                self._cnn_output_size,
                activation='selu',
                bias_initializer=tf.constant_initializer(0.1),
                kernel_initializer='lecun_normal'
            )(layer)
            layer = keras.layers.BatchNormalization(
                beta_initializer=tf.constant_initializer(0.1),
                moving_mean_initializer=tf.constant_initializer(0.1),
                # fused=True  #fused mode only support 4D tensors
            )(layer)

        # DNC layers
        for i in range(self._num_dnc_layers):
            output_size = self.output_size[i] if isinstance(
                self.output_size, list) else self.output_size
            forward = keras.layers.RNN(
                cell=dnc.DNC(
                    name='dnc_fwd_{}'.format(i),
                    output_size=output_size,
                    controller_units=self.controller_units,
                    memory_size=self.memory_size,
                    word_size=self.word_size,
                    num_read_heads=self.num_read_heads,
                    layer_norm_lstm=self._layer_norm_lstm
                ),
                return_sequences=True if i+1 < self._num_dnc_layers else False,
                name='rnn_fwd_{}'.format(i),
            )
            backward = keras.layers.RNN(
                cell=dnc.DNC(
                    name='dnc_bwd_{}'.format(i),
                    output_size=output_size,
                    controller_units=self.controller_units,
                    memory_size=self.memory_size,
                    word_size=self.word_size,
                    num_read_heads=self.num_read_heads,
                    layer_norm_lstm=self._layer_norm_lstm
                ),
                go_backwards=True,
                return_sequences=True if i+1 < self._num_dnc_layers else False,
                name='rnn_bwd_{}'.format(i),
            )
            layer = keras.layers.Bidirectional(
                layer=forward,
                backward_layer=backward,
                name='bidir_{}'.format(i)
            )(layer)

        # Dropout
        dropout = DecayedDropoutLayer(
            dropout_type='AlphaDropout',
            initial_dropout_rate=self._dropout_rate,
            decay_start=self._decayed_dropout_start,
            first_decay_steps=self._dropout_decay_steps,
            t_mul=1.05,
            m_mul=0.98,
            alpha=0.007,
            seed=self.seed,
        )

        if self._num_fcn_layers == 0 and self._dropout_rate > 0:
            layer = dropout(layer)

        # FCN layers
        size = self.output_size
        units = (size[len(size)-1] if isinstance(size, list) else size) * 2
        for i in range(self._num_fcn_layers):
            layer = keras.layers.Dense(
                units=units,
                kernel_initializer='lecun_normal',
                bias_initializer=tf.constant_initializer(0.1),
                activation='selu',
                name='dense_{}'.format(i)
            )(layer)
            if i == 0 and self._dropout_rate > 0:
                layer = dropout(layer)
            units = units // 2

        # Output layer
        outputs = keras.layers.Dense(
            units=self._num_classes,
            bias_initializer=tf.constant_initializer(0.1),
            name='output',
        )(layer)

        self.model = keras.Model(inputs=inputs, outputs=outputs)
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
            loss='sparse_categorical_crossentropy',
            # trying to fix 'Inputs to eager execution function cannot be Keras symbolic tensors'
            # ref: https://github.com/tensorflow/probability/issues/519
            experimental_run_tf_function=False,
            metrics=["accuracy"])

        print(self.model.summary())
