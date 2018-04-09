from __future__ import print_function
# from __future__ import absolute_import
# from __future__ import division

import math
import tensorflow as tf
# pylint: disable-msg=E0611
from tensorflow.python.ops import rnn_cell_impl, math_ops, init_ops, array_ops, nn_ops
from tensorflow.python.framework import ops
from tensorflow.python.layers import base as base_layer

_LayerRNNCell = rnn_cell_impl.LayerRNNCell
_BIAS_VARIABLE_NAME = rnn_cell_impl._BIAS_VARIABLE_NAME
_WEIGHTS_VARIABLE_NAME = rnn_cell_impl._WEIGHTS_VARIABLE_NAME


def numLayers(d1, d2=None):
    n1 = 0
    while d1 > 1:
        d1 = math.ceil(d1/2.0)
        n1 += 1
    n2 = 0
    if d2 is not None:
        n2 = numLayers(d2)
    return max(n1, n2)


def conv2d(input, kernel, filters, seq):
    with tf.variable_scope("conv{}".format(seq)):
        h = int(input.get_shape()[1])
        w = int(input.get_shape()[2])
        conv = tf.layers.conv2d(
            inputs=input,
            name="conv_lv{}".format(seq),
            filters=filters,
            kernel_size=kernel,
            kernel_initializer=tf.truncated_normal_initializer(
                stddev=0.01),
            bias_initializer=tf.constant_initializer(0.1),
            padding="SAME",
            reuse=tf.AUTO_REUSE)
        h_stride = 2 if (h > 2 or w == 2) else 1
        w_stride = 2 if (w > 2 or h == 2) else 1
        pool = tf.layers.max_pooling2d(
            name="pool_lv{}".format(seq),
            inputs=conv, pool_size=2, strides=[h_stride, w_stride],
            padding="SAME")
        ln = tf.contrib.layers.layer_norm(
            scope="ln_{}".format(seq),
            inputs=pool,
            reuse=tf.AUTO_REUSE
        )
        print("rnn-conv{}: {}".format(seq, ln.get_shape()))
        return ln


class EGRUCell(_LayerRNNCell):
    """ Enhanced GRUCell, based on:
    Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).

    Args:
      num_units: int, The number of units in the GRU cell.
      activation: Nonlinearity to use.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
       in an existing scope.  If not `True`, and the existing scope already has
       the given variables, an error is raised.
      kernel_initializer: (optional) The initializer to use for the weight and
      projection matrices.
      bias_initializer: (optional) The initializer to use for the bias.
      name: String, the name of the layer. Layers with the same name will
        share weights, but to avoid mistakes we require reuse=True in such
        cases.
    """

    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 name=None):
        super(EGRUCell, self).__init__(_reuse=reuse, name=name)

        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value
        self._gate_kernel = self.add_variable(
            "gates/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + self._num_units, 2 * self._num_units],
            initializer=self._kernel_initializer)
        self._gate_bias = self.add_variable(
            "gates/%s" % _BIAS_VARIABLE_NAME,
            shape=[2 * self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)))
        self._candidate_kernel = self.add_variable(
            "candidate/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + self._num_units, self._num_units],
            initializer=self._kernel_initializer)
        self._candidate_bias = self.add_variable(
            "candidate/%s" % _BIAS_VARIABLE_NAME,
            shape=[self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype)))

        self.built = True

    def call(self, inputs, state):
        """Gated recurrent unit (GRU) with nunits cells."""

        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), self._gate_kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

        value = math_ops.sigmoid(gate_inputs)
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state

        candidate = math_ops.matmul(
            array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        candidate = nn_ops.bias_add(candidate, self._candidate_bias)

        c = self._activation(candidate)
        new_h = u * state + (1 - u) * c
        return new_h, new_h


class GRUCellv2(_LayerRNNCell):
    """ GRUCell with layer normalization, based on:
    Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).

    Args:
      num_units: int, The number of units in the GRU cell.
      activation: Nonlinearity to use.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
       in an existing scope.  If not `True`, and the existing scope already has
       the given variables, an error is raised.
      kernel_initializer: (optional) The initializer to use for the weight and
      projection matrices.
      bias_initializer: (optional) The initializer to use for the bias.
      name: String, the name of the layer. Layers with the same name will
        share weights, but to avoid mistakes we require reuse=True in such
        cases.
    """

    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 name=None):
        super(GRUCellv2, self).__init__(_reuse=reuse, name=name)

        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value
        self._gate_kernel = self.add_variable(
            "gates/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + self._num_units, 2 * self._num_units],
            initializer=self._kernel_initializer)
        self._gate_bias = self.add_variable(
            "gates/%s" % _BIAS_VARIABLE_NAME,
            shape=[2 * self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)))
        self._candidate_kernel = self.add_variable(
            "candidate/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + self._num_units, self._num_units],
            initializer=self._kernel_initializer)
        self._candidate_bias = self.add_variable(
            "candidate/%s" % _BIAS_VARIABLE_NAME,
            shape=[self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype)))

        self.built = True

    def call(self, inputs, state):
        """Gated recurrent unit (GRU) with nunits cells."""

        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), self._gate_kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

        value = math_ops.sigmoid(gate_inputs)
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r = tf.contrib.layers.layer_norm(
            scope="r_ln",
            inputs=r,
            reuse=tf.AUTO_REUSE
        )

        u = tf.contrib.layers.layer_norm(
            scope="u_ln",
            inputs=u,
            reuse=tf.AUTO_REUSE
        )

        r_state = r * state

        candidate = math_ops.matmul(
            array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        candidate = nn_ops.bias_add(candidate, self._candidate_bias)

        c = self._activation(candidate)
        new_h = u * state + (1 - u) * c
        return new_h, new_h


class EGRUCell_V1(_LayerRNNCell):
    """ Enhanced GRUCell, based on
    Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
    Added 2D Convolution and Layer Normalization.

    Args:
      num_units: int, The number of units in the GRU cell.
      height: feature height. Used to transform the input into 2D shape
       suitable for 2D convolution.
      kernel: convolution kernel size.
      activation: Nonlinearity to use.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
       in an existing scope.  If not `True`, and the existing scope already has
       the given variables, an error is raised.
      kernel_initializer: (optional) The initializer to use for the weight and
      projection matrices.
      bias_initializer: (optional) The initializer to use for the bias.
      name: String, the name of the layer. Layers with the same name will
        share weights, but to avoid mistakes we require reuse=True in such
        cases.
    """

    def __init__(self,
                 num_units,
                 shape,
                 kernel,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 name=None):
        super(EGRUCell_V1, self).__init__(_reuse=reuse, name=name)

        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)
        self._shape = shape
        self._kernel = kernel
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    @property
    def depth(self):
        height = self._shape[0]
        width = self._shape[1]
        nlayer = numLayers(height, width)+1
        depth = max(
            2, 2 ** (math.ceil(math.log(max(height, width), 2)))) * (2 ** nlayer)
        return depth

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        # input_depth = inputs_shape[1].value
        self._gate_kernel = self.add_variable(
            "gates/%s" % _WEIGHTS_VARIABLE_NAME,
            # shape=[input_depth + self._num_units, 2 * self._num_units],
            shape=[self.depth, 2 * self._num_units],
            initializer=self._kernel_initializer)
        self._gate_bias = self.add_variable(
            "gates/%s" % _BIAS_VARIABLE_NAME,
            shape=[2 * self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)))
        self._candidate_kernel = self.add_variable(
            "candidate/%s" % _WEIGHTS_VARIABLE_NAME,
            # shape=[input_depth + self._num_units, self._num_units],
            shape=[self.depth, self._num_units],
            initializer=self._kernel_initializer)
        self._candidate_bias = self.add_variable(
            "candidate/%s" % _BIAS_VARIABLE_NAME,
            shape=[self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype)))

        self.built = True

    def call(self, inputs, state):
        """Gated recurrent unit (GRU) with nunits cells."""

        inputs = self.cnn2d(inputs)

        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), self._gate_kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

        gate_inputs = tf.contrib.layers.layer_norm(
            scope="gate_ln",
            inputs=gate_inputs,
            reuse=tf.AUTO_REUSE
        )

        value = math_ops.sigmoid(gate_inputs)
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        # r = tf.contrib.layers.layer_norm(
        #     scope="r/LN",
        #     inputs=r,
        #     reuse=True
        # )
        # u = tf.contrib.layers.layer_norm(
        #     scope="u/LN",
        #     inputs=u,
        #     reuse=True
        # )

        r_state = r * state

        candidate = math_ops.matmul(
            array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        candidate = nn_ops.bias_add(candidate, self._candidate_bias)

        c = self._activation(candidate)
        new_h = u * state + (1 - u) * c
        return new_h, new_h

    def cnn2d(self, input):
        height = self._shape[0]
        width = self._shape[1]
        # Transforms into 2D compatible format [batch(step), height, width, channel]
        input2d = tf.reshape(input, [-1, height, width, 1])
        nlayer = numLayers(height, width)
        filters = max(
            2, 2 ** (math.ceil(math.log(max(height, width), 2))))
        convlayer = input2d
        for i in range(nlayer):
            filters *= 2
            convlayer = conv2d(convlayer, self._kernel, filters, i)
            convlayer = tf.contrib.layers.layer_norm(
                scope="conv_ln_{}".format(i),
                inputs=convlayer,
                reuse=tf.AUTO_REUSE
            )
        convlayer = tf.squeeze(convlayer, [1, 2])
        return convlayer


class EGRUCell_V2(_LayerRNNCell):
    """ Enhanced GRUCell, based on
    Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
    Added 2D Convolution and Layer Normalization.

    Args:
      num_units: int, The number of units in the GRU cell.
      height: feature height. Used to transform the input into 2D shape
       suitable for 2D convolution.
      shape: a 2D shape(height, width) of feature columns that can be transformed for convolution.
      kernel: convolution kernel size.
      activation: Nonlinearity to use.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
       in an existing scope.  If not `True`, and the existing scope already has
       the given variables, an error is raised.
      kernel_initializer: (optional) The initializer to use for the weight and
      projection matrices.
      bias_initializer: (optional) The initializer to use for the bias.
      name: String, the name of the layer. Layers with the same name will
        share weights, but to avoid mistakes we require reuse=True in such
        cases.
    """

    def __init__(self,
                 num_units,
                 shape,
                 kernel,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 training=None,
                 name=None):
        super(EGRUCell_V2, self).__init__(_reuse=reuse, name=name)

        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)
        self._shape = shape
        self._kernel = kernel
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._training = training

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    @property
    def depth(self):
        height = self._shape[0]
        width = self._shape[1]
        nlayer = numLayers(height, width)
        depth = max(
            2, 2 ** (math.ceil(math.log(max(height, width), 2)))) * (2 ** nlayer)
        return depth + self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        # input_depth = inputs_shape[1].value
        self._gate_kernel = self.add_variable(
            "gates/%s" % _WEIGHTS_VARIABLE_NAME,
            # shape=[input_depth + self._num_units, 2 * self._num_units],
            shape=[self.depth, 2 * self._num_units],
            initializer=self._kernel_initializer)
        self._gate_bias = self.add_variable(
            "gates/%s" % _BIAS_VARIABLE_NAME,
            shape=[2 * self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)))
        self._candidate_kernel = self.add_variable(
            "candidate/%s" % _WEIGHTS_VARIABLE_NAME,
            # shape=[input_depth + self._num_units, self._num_units],
            shape=[self.depth, self._num_units],
            initializer=self._kernel_initializer)
        self._candidate_bias = self.add_variable(
            "candidate/%s" % _BIAS_VARIABLE_NAME,
            shape=[self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype)))

        self.built = True

    def call(self, inputs, state):
        """Gated recurrent unit (GRU) with nunits cells."""

        inputs = self.cnn2d(inputs)

        inputs = tf.layers.dropout(
            inputs=inputs, rate=0.5, training=self._training)

        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), self._gate_kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

        gate_inputs = tf.contrib.layers.layer_norm(
            scope="gate_ln",
            inputs=gate_inputs,
            reuse=tf.AUTO_REUSE
        )

        value = math_ops.sigmoid(gate_inputs)
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state

        candidate = math_ops.matmul(
            array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        candidate = nn_ops.bias_add(candidate, self._candidate_bias)

        c = self._activation(candidate)
        new_h = u * state + (1 - u) * c
        return new_h, new_h

    def cnn2d(self, input):
        height = self._shape[0]
        width = self._shape[1]
        # depth = input.get_shape()[1]
        # Transforms into 2D compatible format [batch(step), height, width, channel]
        input2d = tf.reshape(input, [-1, height, width, 1])
        nlayer = numLayers(height, width)
        filters = max(
            2, 2 ** (math.ceil(math.log(max(height, width), 2))))
        convlayer = input2d
        for i in range(nlayer):
            filters *= 2
            convlayer = conv2d(convlayer, self._kernel, int(filters), i)
        convlayer = tf.layers.flatten(convlayer)
        return convlayer


class DenseCellWrapper(tf.nn.rnn_cell.RNNCell):
    """DenseCell wrapper that ensures cell inputs are concatenated to the outputs."""

    def __init__(self, cell, activation=None):
        """Constructs a `DenseCellWrapper` for `cell`.

        Args:
          cell: An instance of `RNNCell`.
          activation: Activation function that will be applied to the concatenated output.
        """
        self._cell = cell
        self._activation_fn = activation

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size + self._cell.input.get_shape()[-1]

    def zero_state(self, batch_size, dtype):
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            return self._cell.zero_state(batch_size, dtype)

    def __call__(self, inputs, state, scope=None):
        """Run the cell and then apply the concatenation on its inputs to its outputs.

        Args:
          inputs: cell inputs.
          state: cell state.
          scope: optional cell scope.

        Returns:
          Tuple of cell outputs and new state.
        """
        outputs, new_state = self._cell(inputs, state, scope=scope)
        concat_outputs = tf.concat([inputs, outputs], -1)
        if self._activation_fn is not None:
            concat_outputs = self._activation_fn(concat_outputs)
        return (concat_outputs, new_state)
