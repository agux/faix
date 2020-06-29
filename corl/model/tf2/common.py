
import tensorflow as tf
from tensorflow import keras
from time import strftime

class DelayedCosineDecayRestarts(keras.experimental.CosineDecayRestarts):

    def __init__(self, decay_start, *args, **kwargs):
        super(DelayedCosineDecayRestarts, self).__init__(*args, **kwargs)
        self._decay_start = decay_start

    def __call__(self, step):
        return tf.cond(
            tf.less(step, self._decay_start), 
            lambda: self.initial_learning_rate,
            lambda: self.decay(step)
        )

    def decay(self, step):
        lr = super(DelayedCosineDecayRestarts, self).__call__(step-self._decay_start+1)
        # tf.print("DelayedCosineDecayRestarts activated at step: ", step, ", lr= ", lr)
        return lr

    def get_config(self):
        config = super(DelayedCosineDecayRestarts, self).get_config().copy()
        config.update({
            "decay_start": self._decay_start
        })
        return config

class DecayedDropoutLayer(keras.layers.Layer):
    def __init__(self, 
        dropout="dropout",
        decay_start=20000,
        initial_dropout_rate=0.5,
        first_decay_steps=1000,
        t_mul=2.0,
        m_mul=1.0,
        alpha=0.0,
        seed=None,
        *args, **kwargs):
        super(DecayedDropoutLayer, self).__init__(*args, **kwargs)
        self.dropout=dropout.lower()
        self.initial_dropout_rate = initial_dropout_rate
        self.first_decay_steps = first_decay_steps
        self._decay_start = decay_start
        self._t_mul = t_mul
        self._m_mul = m_mul
        self.alpha = alpha
        self.seed = seed

    def build(self, input_shape):
        super(DecayedDropoutLayer, self).build(input_shape)
        self.global_step = self.add_weight(initializer="zeros",
                                        dtype=tf.int32,
                                        trainable=False,
                                        name="global_step")
        self.cosine_decay_restarts = keras.experimental.CosineDecayRestarts(
            initial_learning_rate = self.initial_dropout_rate,
            first_decay_steps=self.first_decay_steps,
            t_mul=self._t_mul,
            m_mul=self._m_mul,
            alpha=self.alpha,
            name='cosine_dacay_restarts'
        )
        if self.dropout == 'dropout':
            self.dropout_layer = keras.layers.Dropout(
                rate=self.initial_dropout_rate, 
                seed=self.seed
            )
        elif self.dropout == 'alphadropout':
            self.dropout_layer = keras.layers.AlphaDropout(
                rate=self.initial_dropout_rate, 
                seed=self.seed
            )
        else:
            raise Exception('unsupported dropout type: {}'.format(self.dropout))

    def call(self, inputs, training=None):
        if training is None:
            training = keras.backend.learning_phase()
        output = tf.cond(
            training,
            lambda: self.train(inputs),
            lambda: tf.identity(inputs)
        )
        return output
    
    def train(self, inputs):
        self.global_step.assign_add(1)
        rate = tf.cond(
            tf.less(self.global_step, self._decay_start),
            lambda: self.initial_dropout_rate,
            lambda: self.cosine_decay_restarts(self.global_step-self._decay_start+1)
        )
        self.dropout_layer.rate = rate
        output = self.dropout_layer(inputs)
        tf.print('step: ', self.global_step, ', dropout rate: ', rate)
        return output

    def get_config(self):
        config = super(DecayedDropoutLayer, self).get_config().copy()
        config.update({
            "dropout": self.dropout,
            "decay_start": self._decay_start,
            "initial_dropout_rate": self.initial_dropout_rate,
            "first_decay_steps": self.first_decay_steps,
            "t_mul": self._t_mul,
            "m_mul": self._m_mul,
            "alpha": self.alpha,
            "seed": self.seed
        })
        return config

class CausalConv1D(keras.layers.Layer):
    def __init__(self, num_cnn_layers, filters, kernels, output_size, *args, **kwargs):
        super(CausalConv1D, self).__init__(*args, **kwargs)
        self.num_cnn_layers = num_cnn_layers
        self.filters = filters
        self.kernels = kernels
        self.output_size = output_size

    def build(self, input_shape):
        super(CausalConv1D, self).build(input_shape)
        self.time_step = int(input_shape[1])
        self.feat_size = int(input_shape[2])
        self.cnn_layers = []
        self.bn_layers = []
        self.out_size = []
        filters = self.filters
        kernels = self.kernels
        for i in range(self.num_cnn_layers):
            self.cnn_layers.append(
                keras.layers.TimeDistributed(
                    keras.layers.Conv1D(
                        filters = filters[i] if isinstance(filters, list) else filters,
                        kernel_size= kernels[i] if isinstance(kernels, list) else kernels,
                        padding='causal', 
                        dilation_rate=i+1, 
                        activation='selu', 
                        bias_initializer=tf.constant_initializer(0.1),
                    )
                )
            )
            self.bn_layers.append(
                keras.layers.BatchNormalization(
                    beta_initializer=tf.constant_initializer(0.1),
                    moving_mean_initializer=tf.constant_initializer(0.1),
                    fused=True
                )
            )
            if isinstance(filters, list):
                self.out_size = filters * self.feat_size
            else:
                self.out_size = [filters * self.feat_size for _ in range(self.num_cnn_layers)]

        self.output_layer = keras.layers.Dense(
            self.output_size, 
            activation='tanh', 
            bias_initializer=tf.constant_initializer(0.1),
        )

    def call(self, inputs):
        reshaped = keras.layers.Reshape([self.feat_size, self.time_step, 1])(inputs)
        outputs = []
        for i in range(self.num_cnn_layers):
            outputs.append(
                keras.layers.Reshape([self.time_step, self.out_size[i]])(
                    self.bn_layers[i](self.cnn_layers[i](reshaped))
                )
            )
        out = tf.concat([inputs]+outputs, axis=-1)
        out = self.output_layer(out)
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_cnn_layers': self.num_cnn_layers,
            'filters': self.filters,
            'kernels': self.kernels,
            'output_size': self.output_size
        })
        return config


class CausalConv1D_V2(keras.layers.Layer):
    def __init__(self, num_cnn_layers, filters, kernels, activation=None, *args, **kwargs):
        super(CausalConv1D_V2, self).__init__(*args, **kwargs)
        self.num_cnn_layers = num_cnn_layers
        self.filters = filters
        self.kernels = kernels
        self.activation = activation

    def build(self, input_shape):
        super(CausalConv1D_V2, self).build(input_shape)
        self.time_step = int(input_shape[1])
        self.feat_size = int(input_shape[2])
        self.cnn_layers = []
        self.bn_layers = []
        self.out_size = []
        filters = self.filters
        kernels = self.kernels
        for i in range(self.num_cnn_layers):
            self.cnn_layers.append(
                keras.layers.TimeDistributed(
                    keras.layers.Conv1D(
                        filters = filters[i] if isinstance(filters, list) else filters,
                        kernel_size= kernels[i] if isinstance(kernels, list) else kernels,
                        padding='causal', 
                        dilation_rate=i+1, 
                        activation=self.activation, 
                        bias_initializer=tf.constant_initializer(0.1),
                        kernel_initializer='lecun_normal' if self.activation == 'selu' else 'glorot_uniform'
                    )
                )
            )
            self.bn_layers.append(
                keras.layers.BatchNormalization(
                    beta_initializer=tf.constant_initializer(0.1),
                    moving_mean_initializer=tf.constant_initializer(0.1),
                    fused=True
                )
            )
            if isinstance(filters, list):
                self.out_size = filters * self.feat_size
            else:
                self.out_size = [filters * self.feat_size for _ in range(self.num_cnn_layers)]

    def call(self, inputs):
        reshaped = keras.layers.Reshape([self.feat_size, self.time_step, 1])(inputs)
        outputs = []
        for i in range(self.num_cnn_layers):
            outputs.append(
                keras.layers.Reshape([self.time_step, self.out_size[i]])(
                    self.bn_layers[i](self.cnn_layers[i](reshaped))
                )
            )
        out = tf.concat([inputs]+outputs, axis=-1)
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_cnn_layers': self.num_cnn_layers,
            'filters': self.filters,
            'kernels': self.kernels,
            'activation': self.activation
        })
        return config