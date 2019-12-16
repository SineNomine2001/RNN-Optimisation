from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import warnings

from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.base_layer import Layer
from keras.engine.base_layer import InputSpec
from keras.utils.generic_utils import has_arg
from keras.utils.generic_utils import to_list
from keras.layers.recurrent import RNN, _generate_dropout_mask

# Legacy support.
from keras.legacy.layers import Recurrent
from keras.legacy import interfaces

class SDot(Layer):
    def __init__(self, units,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        super(SDot, self).__init__(**kwargs)

        self.units = units
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True
    
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 3),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        
        # update gate
        self.kernel_z = self.kernel[:, :self.units]
        # forget gate
        self.kernel_f = self.kernel[:, self.units: self.units * 2]
        # reset gate
        self.kernel_r = self.kernel[:, self.units * 2:]

        self.bias = self.add_weight(
            shape=(self.units * 2,),
            name='bias',
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint)

        # forget bias 
        self.bias_f = self.bias[:self.units]
        # reset bias
        self.bias_r = self.bias[self.units:]

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        inputs_z = inputs
        inputs_f = inputs
        inputs_r = inputs
        
        z = K.dot(inputs_z, self.kernel_z)

        f = K.dot(inputs_f, self.kernel_f)
        f = K.bias_add(f, self.bias_f, data_format='channels_last')
        f = activations.sigmoid(f)

        r = K.dot(inputs_r, self.kernel_r)
        r = K.bias_add(r, self.bias_r, data_format='channels_last')
        r = activations.sigmoid(r)

        output = K.concatenate([z, f, r])

        return output
    
    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units * 3

        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }

        base_config = super(SDot, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SRUCell(Layer):
    def __init__(self, units,
                 dropout=0.,
                 **kwargs):

        super(SRUCell, self).__init__(**kwargs)

        self.units = units
        self.dropout = min(1., max(0., dropout))
        self.state_size = self.units
        self.output_size = self.units
        self._dropout_mask = None

    def call(self, inputs, states, training=None):

        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs),
                self.dropout,
                training=training,
                count=3)
        
        # dropout matrices for input units
        dp_mask = self._dropout_mask

        c_tm1 = states[0]  # previous carry state
        
        if 0. < self.dropout < 1.:
            z = inputs[:, :self.units] * dp_mask[0]
            f = inputs[:, self.units: self.units * 2] * dp_mask[1]
            r = inputs[:, self.units * 2:] * dp_mask[2]
        else:
            z = inputs[:, :self.units]
            f = inputs[:, self.units: self.units * 2]
            r = inputs[:, self.units * 2:]

        c = f * c_tm1 + (1 - f) * z
        h = r * activations.tanh(c) + (1 - r) * z

        if 0 < self.dropout:
            if training is None:
                h._uses_learning_phase = True
        
        return h, [c]
    
    def get_config(self):
    
        config = {'units': self.units,
                  'dropout': self.dropout}

        base_config = super(SRUCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SRU(RNN):
    @interfaces.legacy_recurrent_support
    def __init__(self, units,
                 activity_regularizer=None,
                 dropout=0.,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 reset_after=False,
                 **kwargs):

        if K.backend() == 'theano' and (dropout):
            warnings.warn(
                'RNN dropout is no longer supported with the Theano backend '
                'due to technical limitations. '
                'You can either set `dropout` to 0, '
                'or use the TensorFlow backend.')
            dropout = 0.

        cell = SRUCell(units,
                       dropout=dropout)

        super(SRU, self).__init__(cell,
                                  return_sequences=return_sequences,
                                  return_state=return_state,
                                  go_backwards=go_backwards,
                                  stateful=stateful,
                                  unroll=unroll,
                                  **kwargs)

        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        return super(SRU, self).call(inputs,
                                     mask=mask,
                                     training=training,
                                     initial_state=initial_state)
    @property
    def units(self):
        return self.cell.units

    @property
    def dropout(self):
        return self.cell.dropout

    def get_config(self):
        config = {'units': self.units,
                  'activity_regularizer':
                      regularizers.serialize(self.activity_regularizer),
                  'dropout': self.dropout}

        base_config = super(SRU, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)