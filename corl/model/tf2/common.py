
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
        return {
            "decay_start": self._decay_start
        }

class CausalConv1D(keras.layers.Layer):
    def __init__(self, num_cnn_layers, filters, kernels):
        super(CausalConv1D, self).__init__()
        self.num_cnn_layers = num_cnn_layers
        self.filters = filters
        self.kernels = kernels

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
            # self.bn_layers.append(
            #     keras.layers.BatchNormalization(
            #         beta_initializer=tf.constant_initializer(0.1),
            #         moving_mean_initializer=tf.constant_initializer(0.1),
            #         fused=True
            #     )
            # )
            if isinstance(filters, list):
                self.out_size = filters * self.feat_size
            else:
                self.out_size = [filters * self.feat_size for _ in range(self.num_cnn_layers)]

    def call(self, inputs):
        inputs = keras.layers.Reshape([self.feat_size, self.time_step, 1])(inputs)
        outputs = []
        for i in range(self.num_cnn_layers):
            outputs.append(
                keras.layers.Reshape([self.time_step, self.out_size[i]])(
                    self.cnn_layers[i](inputs)
                )
            )
        out = tf.stack(inputs+outputs, axis=-1)
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_cnn_layers': self.num_cnn_layers,
            'filters': self.filters,
            'kernels': self.kernels
        })
        return config