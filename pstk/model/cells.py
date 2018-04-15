from __future__ import print_function
# from __future__ import absolute_import
# from __future__ import division

import math
import hashlib
import numbers
import tensorflow as tf
# pylint: disable-msg=E0611
from tensorflow.python.ops import rnn_cell_impl, math_ops, init_ops, array_ops, nn_ops, random_ops
from tensorflow.python.framework import ops, tensor_util, tensor_shape
from tensorflow.python.layers import base as base_layer
from tensorflow.python.util import nest

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


class LayerNormGRUCell(_LayerRNNCell):
    """Based on Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
      With Layer Normalization.

    Args:
      num_units: int, The number of units in the GRU cell.
      activation: Nonlinearity to use.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
       in an existing scope.  If not `True`, and the existing scope already has
       the given variables, an error is raised.
      kernel_initializer: (optional) The initializer to use for the weight and
      projection matrices.
      bias_initializer: (optional) The initializer to use for the bias.
      layer_norm: bool, whether to use layer normalization
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
                 layer_norm=False,
                 name=None):
        super(LayerNormGRUCell, self).__init__(_reuse=reuse, name=name)

        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._layer_norm = layer_norm

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

        if self._layer_norm:
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


class LayerNormNASCell(tf.nn.rnn_cell.RNNCell):
    """Neural Architecture Search (NAS) recurrent network cell, with Layer Normalization.

    This implements the recurrent cell from the paper:

      https://arxiv.org/abs/1611.01578

    Barret Zoph and Quoc V. Le.
    "Neural Architecture Search with Reinforcement Learning" Proc. ICLR 2017.

    The class uses an optional projection layer.
    """

    def __init__(self, num_units, num_proj=None, use_biases=False, layer_norm=False, reuse=None):
        """Initialize the parameters for a NAS cell.

        Args:
          num_units: int, The number of units in the NAS cell
          num_proj: (optional) int, The output dimensionality for the projection
            matrices.  If None, no projection is performed.
          use_biases: (optional) bool, If True then use biases within the cell. This
            is False by default.
          layer_norm: (optional) bool, whether to use layer normalization.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.
        """
        super(LayerNormNASCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._num_proj = num_proj
        self._use_biases = use_biases
        self._layer_norm = layer_norm
        self._reuse = reuse

        if num_proj is not None:
            self._state_size = rnn_cell_impl.LSTMStateTuple(
                num_units, num_proj)
            self._output_size = num_proj
        else:
            self._state_size = rnn_cell_impl.LSTMStateTuple(
                num_units, num_units)
            self._output_size = num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def call(self, inputs, state):
        """Run one step of NAS Cell.

        Args:
          inputs: input Tensor, 2D, batch x num_units.
          state: This must be a tuple of state Tensors, both `2-D`, with column
            sizes `c_state` and `m_state`.

        Returns:
          A tuple containing:
          - A `2-D, [batch x output_dim]`, Tensor representing the output of the
            NAS Cell after reading `inputs` when previous state was `state`.
            Here output_dim is:
               num_proj if num_proj was set,
               num_units otherwise.
          - Tensor(s) representing the new state of NAS Cell after reading `inputs`
            when the previous state was `state`.  Same type and shape(s) as `state`.

        Raises:
          ValueError: If input size cannot be inferred from inputs via
            static shape inference.
        """
        sigmoid = math_ops.sigmoid
        tanh = math_ops.tanh
        selu = tf.nn.selu

        num_proj = self._num_units if self._num_proj is None else self._num_proj

        (c_prev, m_prev) = state

        dtype = inputs.dtype
        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError(
                "Could not infer input size from inputs.get_shape()[-1]")
        # Variables for the NAS cell. W_m is all matrices multiplying the
        # hidden state and W_inputs is all matrices multiplying the inputs.
        concat_w_m = tf.variable_scope.get_variable("recurrent_kernel",
                                                    [num_proj, 8 * self._num_units], dtype)
        concat_w_inputs = tf.variable_scope.get_variable(
            "kernel", [input_size.value, 8 * self._num_units], dtype)

        m_matrix = math_ops.matmul(m_prev, concat_w_m)
        inputs_matrix = math_ops.matmul(inputs, concat_w_inputs)

        if self._use_biases:
            b = tf.variable_scope.get_variable(
                "bias",
                shape=[8 * self._num_units],
                initializer=init_ops.zeros_initializer(),
                dtype=dtype)
            m_matrix = nn_ops.bias_add(m_matrix, b)

        if self._layer_norm:
            m_matrix = tf.contrib.layers.layer_norm(
                scope="m_matrix_ln",
                inputs=m_matrix,
                reuse=tf.AUTO_REUSE
            )
            inputs_matrix = tf.contrib.layers.layer_norm(
                scope="inputs_matrix_ln",
                inputs=inputs_matrix,
                reuse=tf.AUTO_REUSE
            )

        # The NAS cell branches into 8 different splits for both the hidden state
        # and the input
        m_matrix_splits = array_ops.split(
            axis=1, num_or_size_splits=8, value=m_matrix)
        inputs_matrix_splits = array_ops.split(
            axis=1, num_or_size_splits=8, value=inputs_matrix)

        # First layer
        layer1_0 = sigmoid(inputs_matrix_splits[0] + m_matrix_splits[0])
        layer1_1 = selu(inputs_matrix_splits[1] + m_matrix_splits[1])
        layer1_2 = sigmoid(inputs_matrix_splits[2] + m_matrix_splits[2])
        layer1_3 = selu(inputs_matrix_splits[3] * m_matrix_splits[3])
        layer1_4 = tanh(inputs_matrix_splits[4] + m_matrix_splits[4])
        layer1_5 = sigmoid(inputs_matrix_splits[5] + m_matrix_splits[5])
        layer1_6 = tanh(inputs_matrix_splits[6] + m_matrix_splits[6])
        layer1_7 = sigmoid(inputs_matrix_splits[7] + m_matrix_splits[7])

        # Second layer
        l2_0 = tanh(layer1_0 * layer1_1)
        l2_1 = tanh(layer1_2 + layer1_3)
        l2_2 = tanh(layer1_4 * layer1_5)
        l2_3 = sigmoid(layer1_6 + layer1_7)

        # Inject the cell
        l2_0 = tanh(l2_0 + c_prev)

        # Third layer
        l3_0_pre = l2_0 * l2_1
        new_c = l3_0_pre  # create new cell
        l3_0 = l3_0_pre
        l3_1 = tanh(l2_2 + l2_3)

        # Final layer
        new_m = tanh(l3_0 * l3_1)

        # Projection layer if specified
        if self._num_proj is not None:
            concat_w_proj = tf.variable_scope.get_variable("projection_weights",
                                                           [self._num_units, self._num_proj], dtype)
            new_m = math_ops.matmul(new_m, concat_w_proj)

        new_state = rnn_cell_impl.LSTMStateTuple(new_c, new_m)
        return new_m, new_state


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

# pylint: disable-msg=E1101
class AlphaDropoutWrapper(tf.nn.rnn_cell.RNNCell):
    """Operator adding alpha dropout to inputs/states/outputs of the given cell."""

    def __init__(self, cell, input_keep_prob=1.0, output_keep_prob=1.0,
                 state_keep_prob=1.0, variational_recurrent=False,
                 input_size=None, dtype=None, seed=None,
                 dropout_state_filter_visitor=None):
        """Create a cell with added input, state, and/or output dropout.

        If `variational_recurrent` is set to `True` (**NOT** the default behavior),
        then the same dropout mask is applied at every step, as described in:

        Y. Gal, Z Ghahramani.  "A Theoretically Grounded Application of Dropout in
        Recurrent Neural Networks".  https://arxiv.org/abs/1512.05287

        Otherwise a different dropout mask is applied at every time step.

        Note, by default (unless a custom `dropout_state_filter` is provided),
        the memory state (`c` component of any `LSTMStateTuple`) passing through
        a `DropoutWrapper` is never modified.  This behavior is described in the
        above article.

        Args:
          cell: an RNNCell, a projection to output_size is added to it.
          input_keep_prob: unit Tensor or float between 0 and 1, input keep
            probability; if it is constant and 1, no input dropout will be added.
          output_keep_prob: unit Tensor or float between 0 and 1, output keep
            probability; if it is constant and 1, no output dropout will be added.
          state_keep_prob: unit Tensor or float between 0 and 1, output keep
            probability; if it is constant and 1, no output dropout will be added.
            State dropout is performed on the outgoing states of the cell.
            **Note** the state components to which dropout is applied when
            `state_keep_prob` is in `(0, 1)` are also determined by
            the argument `dropout_state_filter_visitor` (e.g. by default dropout
            is never applied to the `c` component of an `LSTMStateTuple`).
          variational_recurrent: Python bool.  If `True`, then the same
            dropout pattern is applied across all time steps per run call.
            If this parameter is set, `input_size` **must** be provided.
          input_size: (optional) (possibly nested tuple of) `TensorShape` objects
            containing the depth(s) of the input tensors expected to be passed in to
            the `DropoutWrapper`.  Required and used **iff**
             `variational_recurrent = True` and `input_keep_prob < 1`.
          dtype: (optional) The `dtype` of the input, state, and output tensors.
            Required and used **iff** `variational_recurrent = True`.
          seed: (optional) integer, the randomness seed.
          dropout_state_filter_visitor: (optional), default: (see below).  Function
            that takes any hierarchical level of the state and returns
            a scalar or depth=1 structure of Python booleans describing
            which terms in the state should be dropped out.  In addition, if the
            function returns `True`, dropout is applied across this sublevel.  If
            the function returns `False`, dropout is not applied across this entire
            sublevel.
            Default behavior: perform dropout on all terms except the memory (`c`)
            state of `LSTMCellState` objects, and don't try to apply dropout to
            `TensorArray` objects:
            ```
            def dropout_state_filter_visitor(s):
              if isinstance(s, LSTMCellState):
                # Never perform dropout on the c state.
                return LSTMCellState(c=False, h=True)
              elif isinstance(s, TensorArray):
                return False
              return True
            ```

        Raises:
          TypeError: if `cell` is not an `RNNCell`, or `keep_state_fn` is provided
            but not `callable`.
          ValueError: if any of the keep_probs are not between 0 and 1.
        """
        if not rnn_cell_impl._like_rnncell(cell):
            raise TypeError("The parameter cell is not a RNNCell.")
        if (dropout_state_filter_visitor is not None
                and not callable(dropout_state_filter_visitor)):
            raise TypeError("dropout_state_filter_visitor must be callable")
        self._dropout_state_filter = (
            dropout_state_filter_visitor or rnn_cell_impl._default_dropout_state_filter_visitor)
        with ops.name_scope("DropoutWrapperInit"):
            def tensor_and_const_value(v):
                tensor_value = ops.convert_to_tensor(v)
                const_value = tensor_util.constant_value(tensor_value)
                return (tensor_value, const_value)
            for prob, attr in [(input_keep_prob, "input_keep_prob"),
                               (state_keep_prob, "state_keep_prob"),
                               (output_keep_prob, "output_keep_prob")]:
                tensor_prob, const_prob = tensor_and_const_value(prob)
                if const_prob is not None:
                    if const_prob < 0 or const_prob > 1:
                        raise ValueError("Parameter %s must be between 0 and 1: %d"
                                         % (attr, const_prob))
                    setattr(self, "_%s" % attr, float(const_prob))
                else:
                    setattr(self, "_%s" % attr, tensor_prob)

        # Set cell, variational_recurrent, seed before running the code below
        self._cell = cell
        self._variational_recurrent = variational_recurrent
        self._seed = seed

        self._recurrent_input_noise = None
        self._recurrent_state_noise = None
        self._recurrent_output_noise = None

        if variational_recurrent:
            if dtype is None:
                raise ValueError(
                    "When variational_recurrent=True, dtype must be provided")

            def convert_to_batch_shape(s):
                # Prepend a 1 for the batch dimension; for recurrent
                # variational dropout we use the same dropout mask for all
                # batch elements.
                return array_ops.concat(
                    ([1], tensor_shape.TensorShape(s).as_list()), 0)

            def batch_noise(s, inner_seed):
                shape = convert_to_batch_shape(s)
                return random_ops.random_uniform(shape, seed=inner_seed, dtype=dtype)

            if (not isinstance(self._input_keep_prob, numbers.Real) or
                    self._input_keep_prob < 1.0):
                if input_size is None:
                    raise ValueError(
                        "When variational_recurrent=True and input_keep_prob < 1.0 or "
                        "is unknown, input_size must be provided")
                self._recurrent_input_noise = rnn_cell_impl._enumerated_map_structure_up_to(
                    input_size,
                    lambda i, s: batch_noise(
                        s, inner_seed=self._gen_seed("input", i)),
                    input_size)
            self._recurrent_state_noise = rnn_cell_impl._enumerated_map_structure_up_to(
                cell.state_size,
                lambda i, s: batch_noise(
                    s, inner_seed=self._gen_seed("state", i)),
                cell.state_size)
            self._recurrent_output_noise = rnn_cell_impl._enumerated_map_structure_up_to(
                cell.output_size,
                lambda i, s: batch_noise(
                    s, inner_seed=self._gen_seed("output", i)),
                cell.output_size)

    def _gen_seed(self, salt_prefix, index):
        if self._seed is None:
            return None
        salt = "%s_%d" % (salt_prefix, index)
        string = (str(self._seed) + salt).encode("utf-8")
        return int(hashlib.md5(string).hexdigest()[:8], 16) & 0x7FFFFFFF

    @property
    def wrapped_cell(self):
        return self._cell

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def zero_state(self, batch_size, dtype):
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            return self._cell.zero_state(batch_size, dtype)

    def _variational_recurrent_dropout_value(
            self, index, value, noise, keep_prob):
        """Performs dropout given the pre-calculated noise tensor."""
        # uniform [keep_prob, 1.0 + keep_prob)
        random_tensor = keep_prob + noise

        # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        binary_tensor = math_ops.floor(random_tensor)
        ret = math_ops.div(value, keep_prob) * binary_tensor
        ret.set_shape(value.get_shape())
        return ret

    def _dropout(self, values, salt_prefix, recurrent_noise, keep_prob,
                 shallow_filtered_substructure=None):
        """Decides whether to perform alpha dropout or recurrent dropout."""

        if shallow_filtered_substructure is None:
            # Put something so we traverse the entire structure; inside the
            # dropout function we check to see if leafs of this are bool or not.
            shallow_filtered_substructure = values

        if not self._variational_recurrent:
            def dropout(i, do_dropout, v):
                if not isinstance(do_dropout, bool) or do_dropout:
                    return tf.contrib.nn.alpha_dropout(
                        x=v,
                        keep_prob=keep_prob,
                        seed=self._gen_seed(salt_prefix, i)
                    )
                else:
                    return v
            return rnn_cell_impl._enumerated_map_structure_up_to(
                shallow_filtered_substructure, dropout,
                *[shallow_filtered_substructure, values])
        else:
            def dropout(i, do_dropout, v, n):
                if not isinstance(do_dropout, bool) or do_dropout:
                    return self._variational_recurrent_dropout_value(i, v, n, keep_prob)
                else:
                    return v
            return rnn_cell_impl._enumerated_map_structure_up_to(
                shallow_filtered_substructure, dropout,
                *[shallow_filtered_substructure, values, recurrent_noise])

    def __call__(self, inputs, state, scope=None):
        """Run the cell with the declared dropouts."""
        def _should_dropout(p):
            return (not isinstance(p, float)) or p < 1

        if _should_dropout(self._input_keep_prob):
            inputs = self._dropout(inputs, "input",
                                   self._recurrent_input_noise,
                                   self._input_keep_prob)
        output, new_state = self._cell(inputs, state, scope=scope)
        if _should_dropout(self._state_keep_prob):
            # Identify which subsets of the state to perform dropout on and
            # which ones to keep.
            shallow_filtered_substructure = nest.get_traverse_shallow_structure(
                self._dropout_state_filter, new_state)
            new_state = self._dropout(new_state, "state",
                                      self._recurrent_state_noise,
                                      self._state_keep_prob,
                                      shallow_filtered_substructure)
        if _should_dropout(self._output_keep_prob):
            output = self._dropout(output, "output",
                                   self._recurrent_output_noise,
                                   self._output_keep_prob)
        return output, new_state
