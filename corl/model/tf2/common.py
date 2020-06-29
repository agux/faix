
import tensorflow as tf
from tensorflow import keras
from time import strftime

from tensorflow.python.keras.utils import tf_utils

class AlphaDropout(keras.layers.Layer):
  """Applies Alpha Dropout to the input.

  Alpha Dropout is a `Dropout` that keeps mean and variance of inputs
  to their original values, in order to ensure the self-normalizing property
  even after this dropout.
  Alpha Dropout fits well to Scaled Exponential Linear Units
  by randomly setting activations to the negative saturation value.

  Arguments:
    rate: float, drop probability (as with `Dropout`).
      The multiplicative noise will have
      standard deviation `sqrt(rate / (1 - rate))`.
    seed: A Python integer to use as random seed.

  Call arguments:
    inputs: Input tensor (of any rank).
    training: Python boolean indicating whether the layer should behave in
      training mode (adding dropout) or in inference mode (doing nothing).

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.

  Output shape:
    Same shape as input.
  """

  def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
    super(AlphaDropout, self).__init__(**kwargs)
    self.rate = rate
    self.noise_shape = noise_shape
    self.seed = seed
    self.supports_masking = True

  def _get_noise_shape(self, inputs):
    return self.noise_shape if self.noise_shape else tf.shape(inputs)

  def call(self, inputs, training=None):
    def dropped_inputs(inputs=inputs, rate=self.rate, seed=self.seed):  # pylint: disable=missing-docstring
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        alpha_p = -alpha * scale
        kept_idx = tf.math.greater_equal(
            keras.backend.random_uniform(noise_shape, seed=seed), rate)
        kept_idx = tf.cast(kept_idx, inputs.dtype)
        # Get affine transformation params
        a = ((1 - rate) * (1 + rate * alpha_p**2))**-0.5
        b = -a * alpha_p * rate
        # Apply mask
        x = inputs * kept_idx + alpha_p * (1 - kept_idx)
        # Do affine transformation
        return a * x + b
    def dropout():
        noise_shape = self._get_noise_shape(inputs)
        return keras.backend.in_train_phase(dropped_inputs, inputs, training=training)
    return tf_utils.smart_cond(
        tf.math.logical_and(
            tf.math.greater(self.rate, 0.),
            tf.math.less(self.rate, 1.)
        ),
        dropout,
        lambda: tf.identity(inputs)
    )

  def get_config(self):
    config = {'rate': self.rate}
    base_config = super(AlphaDropout, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    return input_shape

class DelayedCosineDecayRestarts(keras.experimental.CosineDecayRestarts):

    def __init__(self, decay_start, *args, **kwargs):
        super(DelayedCosineDecayRestarts, self).__init__(*args, **kwargs)
        self._decay_start = decay_start

    def __call__(self, step):
        return tf.cond(
            tf.math.less(step, self._decay_start), 
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
        dropout_type='dropout'
        decay_start=20000,
        initial_dropout_rate=0.5,
        first_decay_steps=1000,
        t_mul=2.0,
        m_mul=1.0,
        alpha=0.0,
        seed=None,
        *args, **kwargs):
        # kwargs['dynamic'] = True
        super(DecayedDropoutLayer, self).__init__(*args, **kwargs)
        self.dropout_type=dropout_type.lower()
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
        if self.dropout_type == 'dropout':
            self.dropout_layer = keras.layers.Dropout(
                rate=self.initial_dropout_rate, 
                seed=self.seed
            )
        elif self.dropout_type == 'alphadropout':
            self.dropout_layer = AlphaDropout(
                rate=self.initial_dropout_rate, 
                seed=self.seed
            )
        else:
            raise Exception('unsupported dropout type: {}'.format(self.dropout)) 
    
    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, training=None):
        if training is None:
            training = keras.backend.learning_phase()

        def dropout():
            self.global_step.assign_add(1)
            rate = tf_utils.smart_cond(
                tf.math.less(self.global_step, self._decay_start),
                lambda: self.initial_dropout_rate,
                lambda: self.cosine_decay_restarts(self.global_step-self._decay_start+1)
            )
            self.dropout_layer.rate = rate
            return self.dropout_layer(inputs, training)

        output = tf_utils.smart_cond(
            training,
            dropout,
            lambda: tf.identity(inputs)
        )
        return output
    
    def get_config(self):
        config = super(DecayedDropoutLayer, self).get_config().copy()
        config.update({
            "dropout_type": self.dropout_type,
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