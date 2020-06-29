import tensorflow as tf
import numpy as np

from tensorflow import keras
from time import strftime
from corl.model.tf2.common import DelayedCosineDecayRestarts, CausalConv1D, CausalConv1D_V2, DecayedDropoutLayer
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
    '''
        2 Bidirectional DNC layers, dropout and output layer
    '''

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
    '''
        Multiple Bidirectional DNC layers, AlphaDropout, Multiple FCN layers
    '''

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
        for i in range(self._num_fcn_layers):
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

class DNC_Model_V4(DNC_Model):
    '''
        DNC with LayerNormLSTMCell as controller.
    '''

    def __init__(self, num_dnc_layers=2, num_fcn_layers=2, *args, **kwargs):
        super(DNC_Model_V4, self).__init__(*args, **kwargs)
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
                    name='dnc_{}'.format(i),
                    output_size=self.output_size,
                    controller_units=self.controller_units,
                    memory_size=self.memory_size,
                    word_size=self.word_size,
                    num_read_heads=self.num_read_heads,
                    layer_norm_lstm=True
                ),
                return_sequences=True if i+1 < self._num_dnc_layers else False,
                name='rnn_{}'.format(i),
            )
            layer = keras.layers.Bidirectional(layer=rnn,name='bidir_{}'.format(i))(layer)
        
        layer = keras.layers.AlphaDropout(self._dropout_rate)(layer)

        # create sequence of FCN layers
        units = self.output_size
        for i in range(self._num_fcn_layers):
            layer = keras.layers.Dense(
                units=units,
                bias_initializer=tf.constant_initializer(0.1),
                activation='selu',
                name='dense_{}'.format(i)
            )(layer)
            units = units // 2

        # Output layer
        outputs = keras.layers.Dense(
            units=1,
            bias_initializer=tf.constant_initializer(0.1),
            name='output',
        )(layer)

        self.model = keras.Model(inputs=feat, outputs=outputs)
        self.model._name = self.getName()

        return self.model

class DNC_Model_V5(DNC_Model):
    '''
        Multiple Bidirectional DNC layers, AlphaDropout, Multiple FCN layers
    '''

    def __init__(self, num_dnc_layers=2, num_fcn_layers=2, layer_norm_lstm=False, *args, **kwargs):
        super(DNC_Model_V5, self).__init__(*args, **kwargs)
        self._num_dnc_layers = num_dnc_layers
        self._num_fcn_layers = num_fcn_layers
        self._layer_norm_lstm = layer_norm_lstm

    def getModel(self):
        if self.model is not None:
            return self.model
        print('{} constructing model {}'.format(strftime("%H:%M:%S"),
                                                self.getName()))

        inputs = keras.Input(
            shape=(self._time_step, self._feat_size),
            # name='features',
            dtype=tf.float32)

        #TODO add CNN before RNN?

        # create sequence of DNC layers
        layer = inputs
        for i in range(self._num_dnc_layers):
            rnn = keras.layers.RNN(
                cell = dnc.DNC(
                    name='dnc_{}'.format(i),
                    output_size=self.output_size,
                    controller_units=self.controller_units,
                    memory_size=self.memory_size,
                    word_size=self.word_size,
                    num_read_heads=self.num_read_heads,
                    layer_norm_lstm=self._layer_norm_lstm
                ),
                return_sequences=True if i+1 < self._num_dnc_layers else False,
                name='rnn_{}'.format(i),
            )
            # TODO use separate dnc cell for forward & backward pass?
            layer = keras.layers.Bidirectional(layer=rnn, name='bidir_{}'.format(i))(layer)
        
        # TODO add batch normalization layer before FCN?

        if self._dropout_rate > 0:
            layer = keras.layers.AlphaDropout(self._dropout_rate)(layer)

        # create sequence of FCN layers
        units = self.output_size
        for i in range(self._num_fcn_layers):
            layer = keras.layers.Dense(
                units=units,
                bias_initializer=tf.constant_initializer(0.1),
                activation='selu',
                name='dense_{}'.format(i)
            )(layer)
            units = units // 2

        # Output layer
        outputs = keras.layers.Dense(
            units=1,
            bias_initializer=tf.constant_initializer(0.1),
            name='output',
        )(layer)

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model._name = self.getName()

        return self.model

class DNC_Model_V6(DNC_Model):
    '''
        Multiple causal Conv1D layers, Multiple Bidirectional DNC layers, AlphaDropout, Multiple FCN layers
    '''

    def __init__(self, 
        num_cnn_layers=1, 
        num_dnc_layers=2, 
        num_fcn_layers=2, 
        cnn_filters=64, #can be a list
        cnn_kernel_size=3, #can be a list
        layer_norm_lstm=False,
        *args, **kwargs):
        super(DNC_Model_V6, self).__init__(*args, **kwargs)
        self._num_cnn_layers = num_cnn_layers
        self._num_dnc_layers = num_dnc_layers
        self._num_fcn_layers = num_fcn_layers
        self._cnn_filters=cnn_filters
        self._cnn_kernel_size=cnn_kernel_size
        self._layer_norm_lstm = layer_norm_lstm

    def getModel(self):
        if self.model is not None:
            return self.model
        print('{} constructing model {}'.format(strftime("%H:%M:%S"),
                                                self.getName()))

        inputs = keras.Input(
            shape=(self._time_step, self._feat_size),
            # name='features',
            dtype=tf.float32)

        layer = inputs
        # add CNN before RNN
        if self._num_cnn_layers > 0:
            layer = keras.layers.Reshape([self._feat_size, self._time_step, 1])(layer)
            filters = self._cnn_filters
            kernels = self._cnn_kernel_size
            for i in range(self._num_cnn_layers):
                layer = keras.layers.TimeDistributed(
                    keras.layers.Conv1D(
                        filters = filters[i] if isinstance(filters, list) else filters,
                        kernel_size= kernels[i] if isinstance(kernels, list) else kernels,
                        padding='causal', 
                        dilation_rate=i+1, 
                        activation='selu', 
                        bias_initializer=tf.constant_initializer(0.1),
                    )
                )(layer)
            layer = keras.layers.BatchNormalization(
                beta_initializer=tf.constant_initializer(0.1),
                moving_mean_initializer=tf.constant_initializer(0.1),
                fused=True
            )(layer)
            last_dim = filters[len(filters)-1] if isinstance(filters, list) else filters
            layer = keras.layers.Reshape([self._time_step, self._feat_size * last_dim])(layer)
            
        # create sequence of DNC layers
        for i in range(self._num_dnc_layers):
            forward = keras.layers.RNN(
                cell = dnc.DNC(
                    name='dnc_fwd_{}'.format(i),
                    output_size=self.output_size,
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
                cell = dnc.DNC(
                    name='dnc_bwd_{}'.format(i),
                    output_size=self.output_size,
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
            layer = keras.layers.Bidirectional(layer=forward, backward_layer=backward, name='bidir_{}'.format(i))(layer)
        
        if self._dropout_rate > 0:
            layer = keras.layers.AlphaDropout(self._dropout_rate)(layer)

        # create sequence of FCN layers
        units = self.output_size
        for i in range(self._num_fcn_layers):
            layer = keras.layers.Dense(
                units=units,
                bias_initializer=tf.constant_initializer(0.1),
                activation='selu',
                name='dense_{}'.format(i)
            )(layer)
            units = units // 2

        # Output layer
        outputs = keras.layers.Dense(
            units=1,
            bias_initializer=tf.constant_initializer(0.1),
            name='output',
        )(layer)

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model._name = self.getName()

        return self.model

class DNC_Model_V7(DNC_Model):
    '''
        Multiple causal Conv1D layers on same input, BatchNorm, AlphaDropout, Multiple Bidirectional DNC layers, Multiple FCN layers
    '''

    def __init__(self, 
        num_cnn_layers=1, 
        num_dnc_layers=2, 
        num_fcn_layers=2, 
        cnn_filters=64, #can be a list
        cnn_kernel_size=3, #can be a list
        cnn_output_size=256,
        layer_norm_lstm=False,
        *args, **kwargs):
        super(DNC_Model_V7, self).__init__(*args, **kwargs)
        self._num_cnn_layers = num_cnn_layers
        self._num_dnc_layers = num_dnc_layers
        self._num_fcn_layers = num_fcn_layers
        self._cnn_filters = cnn_filters
        self._cnn_kernel_size = cnn_kernel_size
        self._cnn_output_size = cnn_output_size
        self._layer_norm_lstm = layer_norm_lstm
    
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
        # add CNN before RNN
        if self._num_cnn_layers > 0:
            layer = CausalConv1D(
                self._num_cnn_layers, 
                self._cnn_filters, 
                self._cnn_kernel_size,
                self._cnn_output_size
            )(layer)
            layer = keras.layers.BatchNormalization(
                beta_initializer=tf.constant_initializer(0.1),
                moving_mean_initializer=tf.constant_initializer(0.1),
                # fused=True  #fused mode only support 4D tensors
            )(layer)
            if self._dropout_rate > 0:
                layer = keras.layers.AlphaDropout(self._dropout_rate)(layer)
            
        # create sequence of DNC layers
        for i in range(self._num_dnc_layers):
            output_size = self.output_size[i] if isinstance(self.output_size, list) else self.output_size
            forward = keras.layers.RNN(
                cell = dnc.DNC(
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
                cell = dnc.DNC(
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
            layer = keras.layers.Bidirectional(layer=forward, backward_layer=backward, name='bidir_{}'.format(i))(layer)
        
        # create sequence of FCN layers
        units = self.output_size
        for i in range(self._num_fcn_layers):
            layer = keras.layers.Dense(
                units=units,
                bias_initializer=tf.constant_initializer(0.1),
                activation='selu',
                name='dense_{}'.format(i)
            )(layer)
            units = units // 2

        # Output layer
        outputs = keras.layers.Dense(
            units=1,
            bias_initializer=tf.constant_initializer(0.1),
            name='output',
        )(layer)

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model._name = self.getName()

        return self.model

class DNC_Model_V8(DNC_Model_V7):
    '''

        Multiple causal Conv1D layers on same input, BatchNorm, Multiple Bidirectional DNC layers, (DecayedDropout#1), Multiple FCN layers (DecayedDropout#2)
        
        * DecayedDropout#1 exists only if number of FCN layer is 0
        * DecayedDropout#2 exists only if number of FCN layer is > 0

    '''

    def __init__(self, seed, *args, **kwargs):
        super(DNC_Model_V8, self).__init__(*args, **kwargs)
        self.seed = seed

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
                'selu'
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
            output_size = self.output_size[i] if isinstance(self.output_size, list) else self.output_size
            forward = keras.layers.RNN(
                cell = dnc.DNC(
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
                cell = dnc.DNC(
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
        if self._num_fcn_layers == 0 and self._dropout_rate > 0:
            layer = DecayedDropoutLayer(
                dropout="alphadropout",
                decay_start=self._decayed_dropout_start,
                initial_dropout_rate=self._dropout_rate,
                first_decay_steps=self._dropout_decay_steps,
                t_mul=1.05,
                m_mul=0.98,
                alpha=0.007,
                seed=self.seed
            )(layer)

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
                layer = DecayedDropoutLayer(
                    dropout="alphadropout",
                    decay_start=self._decayed_dropout_start,
                    initial_dropout_rate=self._dropout_rate,
                    first_decay_steps=self._dropout_decay_steps,
                    t_mul=1.05,
                    m_mul=0.98,
                    alpha=0.007,
                    seed=self.seed
                )(layer)
            units = units // 2

        # Output layer
        outputs = keras.layers.Dense(
            units=1,
            bias_initializer=tf.constant_initializer(0.1),
            name='output',
        )(layer)

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model._name = self.getName()

        return self.model