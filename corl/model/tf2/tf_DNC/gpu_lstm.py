
import tensorflow as tf
import tensorflow.keras.activations as activations
import tf.math as math

_CUDNN_AVAILABLE_MSG = 'Layer %s will use cuDNN kernel when run on GPU.'
_CUDNN_NOT_AVAILABLE_MSG = ('Layer %s will not use cuDNN kernel since it '
                            'doesn\'t meet the cuDNN kernel criteria. It will '
                            'use generic GPU kernel as fallback when running '
                            'on GPU')

class GPU_LSTM(tf.keras.layers.LSTMCell):

    def __init__(self, units, activation='tanh', recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=2, **kwargs):
        super().__init__(units, activation=activation, recurrent_activation=recurrent_activation, use_bias=use_bias, kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer, bias_initializer=bias_initializer, unit_forget_bias=unit_forget_bias, kernel_regularizer=kernel_regularizer, recurrent_regularizer=recurrent_regularizer, bias_regularizer=bias_regularizer, kernel_constraint=kernel_constraint, recurrent_constraint=recurrent_constraint, bias_constraint=bias_constraint, dropout=dropout, recurrent_dropout=recurrent_dropout, implementation=implementation, **kwargs)

        self._could_use_gpu_kernel = (
            self.activation in (activations.tanh, math.tanh) and
            self.recurrent_activation in (activations.sigmoid, math.sigmoid) and
            recurrent_dropout == 0 and use_bias and
            tf.executing_eagerly())
        num_gpu = len(tf.config.list_physical_devices('GPU'))
        if num_gpu > 0:
            # Only show the message when there is GPU available, user will not care
            # about the cuDNN if there isn't any GPU.
            if self._could_use_gpu_kernel:
                print(_CUDNN_AVAILABLE_MSG % self.name)
            else:
                print(_CUDNN_NOT_AVAILABLE_MSG % self.name)
        