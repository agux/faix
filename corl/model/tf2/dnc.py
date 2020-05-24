import tensorflow as tf
import numpy as np

from tensorflow import keras
from time import strftime
from corl.model.tf2.common import DelayedCosineDecayRestarts
from adnc.model.controller_units.controller import get_rnn_cell_list
from adnc.model.memory_units.memory_unit import get_memory_unit


class OutputLayer(keras.layers.Layer):
    """
        Calculates the weighted and activated output of the MANN model
        Args:
            outputs: TF tensor, concatenation of memory units output and controller output

        Returns: TF tensor, predictions
    """
    def __init__(self, output_function, output_size, seed, **kwargs):
        super(OutputLayer, self).__init__(**kwargs)
        self.output_function = output_function
        self.output_size = output_size
        self.seed = seed

    def build(self, input_shape):
        super(OutputLayer, self).build(input_shape)
        self.last_dim = int(input_shape[-1])
        self.weights_concat = self.add_weight(
            name="weights_concat",
            shape=(self.last_dim, self.output_size),
            initializer=tf.keras.initializers.VarianceScaling(
                scale=1.0,
                mode="fan_avg",
                distribution="uniform",
                seed=self.seed)(shape=(self.last_dim, self.output_size),
                                dtype=self.dtype),
            dtype=self.dtype)
        self.bias_merge = self.add_weight(name="bias_merge",
                                          shape=(self.output_size, ),
                                          initializer=tf.zeros_initializer()(
                                              shape=(self.output_size, ),
                                              dtype=self.dtype),
                                          dtype=self.dtype)

    def call(self, inputs):
        output_flat = tf.reshape(inputs, [-1, self.last_dim])
        output_flat = tf.matmul(output_flat,
                                self.weights_concat) + self.bias_merge
        if self.output_function == 'softmax':
            predictions_flat = tf.nn.softmax(output_flat)
        elif self.output_function == 'tanh':
            predictions_flat = tf.tanh(output_flat)
        elif self.output_function == 'linear':
            predictions_flat = output_flat
        else:
            raise UserWarning(
                "Unknown output function, use softmax, tanh or linear")
        predictions = tf.reshape(predictions_flat, [-1, self.output_size])
        return predictions

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'dtype': self.dtype,
            'output_function': self.output_function,
            'output_size': self.output_size,
            'seed': self.seed
        })
        return config


class BidirectionalLayer(keras.layers.Layer):
    """
        Connects bidirectional controller and memory unit and performs scan over sequence
        Args:
            controller_config:      dict, configuration of the controller
            memory_unit_config:     dict, configuration of the memory unit

        Returns:        TF tensor, output sequence

    """
    def __init__(self, controller_config, memory_unit_config, seed, **kwargs):
        super(BidirectionalLayer, self).__init__(**kwargs)
        self.controller_config = controller_config
        self.memory_unit_config = memory_unit_config
        self.seed = seed

    def build(self, input_shape):
        super(BidirectionalLayer, self).build(input_shape)
        with tf.name_scope("controller"):
            list_fw = get_rnn_cell_list(self.controller_config,
                                        name='con_fw',
                                        seed=self.seed,
                                        dtype=self.dtype)
            list_bw = get_rnn_cell_list(self.controller_config,
                                        name='con_bw',
                                        seed=self.seed,
                                        dtype=self.dtype)

        self.cell_fw = keras.layers.StackedRNNCells(list_fw)
        self.cell_bw = keras.layers.StackedRNNCells(list_bw)

        memory_input_size = self.cell_fw.output_size + self.cell_bw.output_size
        self.cell_mu = get_memory_unit(memory_input_size, self.memory_unit_config,
                                       'memory_unit')

    def call(self, inputs):
        # transpose into time-major input
        inputs = tf.transpose(inputs, [1, 0, 2])
        time_step = inputs.get_shape()[0]
        with tf.name_scope("bw") as bw_scope:
            rnn_bw = keras.layers.RNN(cell=self.cell_bw,
                                      time_major=True,
                                      return_sequences=True,
                                      input_shape=(None, time_step),
                                      go_backwards=True)
            output_bw = rnn_bw(inputs)

        batch_size = inputs.get_shape()[0].value
        cell_fw_init_states = self.cell_fw.get_initial_state(batch_size,
                                                             dtype=self.dtype)
        cell_mu_init_states = self.cell_mu.get_initial_state(batch_size,
                                                             dtype=self.dtype)
        output_init = tf.zeros([batch_size, self.cell_mu.output_size],
                               dtype=self.dtype)

        init_states = (output_init, cell_fw_init_states, cell_mu_init_states)
        coupled_inputs = (inputs, output_bw)

        with tf.name_scope("fw") as fw_scope:

            def step(pre_states, coupled_inputs):
                inputs, output_bw = coupled_inputs
                pre_outputs, pre_states_fw, pre_states_mu = pre_states

                controller_inputs = tf.concat([inputs, pre_outputs], axis=-1)
                output_fw, states_fw = self.cell_fw(controller_inputs,
                                                    pre_states_fw)

                mu_inputs = tf.concat([output_fw, output_bw], axis=-1)
                output_mu, states_mu = self.cell_mu(mu_inputs, pre_states_mu)

                return (output_mu, states_fw, states_mu)

            outputs, states_fw, states_mu = tf.scan(step,
                                                    coupled_inputs,
                                                    initializer=init_states,
                                                    parallel_iterations=32)

        return outputs

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'dtype': self.dtype,
            'controller_config': self.controller_config,
            'memory_unit_config': self.memory_unit_config,
            'seed': self.seed
        })
        return config


class MANN_Model():
    def __init__(self,
                 controller_config,
                 memory_unit_config,
                 name='mann',
                 time_step=30,
                 feat_size=None,
                 dropout_rate=0.5,
                 decayed_dropout_start=None,
                 dropout_decay_steps=None,
                 learning_rate=1e-3,
                 decayed_lr_start=None,
                 lr_decay_steps=None,
                 seed=None,
                 dtype=tf.float32):

        self.seed = seed
        self.rng = np.random.RandomState(seed=self.seed)
        self.dtype = dtype

        self.controller_config = controller_config
        self.memory_unit_config = memory_unit_config

        self._time_step = time_step
        self._feat_size = feat_size
        self._dropout_rate = dropout_rate
        self._decayed_dropout_start = decayed_dropout_start
        self._dropout_decay_steps = dropout_decay_steps
        self._lr = learning_rate
        self._decayed_lr_start = decayed_lr_start
        self._lr_decay_steps = lr_decay_steps

        self.name = name

        self.model = None

    def getName(self):
        if self.name is not None:
            return self.name
        else:
            return self.__class__.__name__

    def getModel(self):
        if self.model is not None:
            return self.model
        print('{} constructing model {}'.format(strftime("%H:%M:%S"),
                                                self.getName()))

        feat = keras.Input(
            shape=(self._time_step, self._feat_size),
            name='features',
            dtype='float32')
        seqlens = keras.Input(shape=(1), name='seqlens', dtype='int32')
        inputs = {'features': feat, 'seqlens': seqlens}

        unweighted_outputs = BidirectionalLayer(
            dtype=tf.float32,
            controller_config=self.controller_config,
            memory_unit_config=self.memory_unit_config,
            seed=self.seed)(feat)

        prediction = OutputLayer(dtype=tf.float32,
                                 output_function='linear',
                                 output_size=0,
                                 seed=self.seed)(unweighted_outputs)

        self.model = keras.Model(inputs=inputs, outputs=prediction)
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
            clipvalue=10)
        self.model.compile(
            optimizer=optimizer,
            loss='huber_loss',
            # trying to fix 'Inputs to eager execution function cannot be Keras symbolic tensors'
            # ref: https://github.com/tensorflow/probability/issues/519
            experimental_run_tf_function=False)
        print(self.model.summary())
