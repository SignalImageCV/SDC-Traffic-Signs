"""Common modules to leaders CNN.

Downsampling, sub-filters...
"""

import math
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils


from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import moving_averages

# ==============================================================================
# Tiny Conv2d
# ==============================================================================
@add_arg_scope
def conv2d_tiny(inputs,
                num_outputs,
                rate=1,
                padding='SAME',
                data_format=None,
                activation_fn=nn.relu,
                normalizer_fn=None,
                normalizer_params=None,
                weights_initializer=initializers.xavier_initializer_conv2d(),
                weights_regularizer=None,
                biases_initializer=init_ops.zeros_initializer,
                biases_regularizer=None,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=True,
                scope=None,):
    """Tiny Convolution 2d.
    """
    with variable_scope.variable_scope(scope, 'Conv', [inputs],
                                       reuse=reuse) as sc:
        inputs = ops.convert_to_tensor(inputs)
        dtype = inputs.dtype.base_dtype
        input_rank = inputs.get_shape().ndims
        if input_rank is None:
            raise ValueError('Rank of inputs must be known')
        if input_rank < 3 or input_rank > 5:
            raise ValueError('Rank of inputs is %d, which is not >= 3 and <= 5' %
                             input_rank)
        conv_dims = input_rank - 2

        # First 2x2 convolution.
        # num_outputs_inter = num_outputs
        output = slim.conv2d(inputs, num_outputs, [2, 2], rate=rate,
                             padding='VALID',
                             activation_fn=None,
                             normalizer_fn=normalizer_fn,
                             normalizer_params=normalizer_params,
                             # normalizer_fn=None,
                             # normalizer_params=None,
                             weights_initializer=initializers.xavier_initializer_conv2d(),
                             weights_regularizer=weights_regularizer,
                             biases_initializer=None,
                             # biases_initializer=init_ops.zeros_initializer,
                             biases_regularizer=biases_regularizer,
                             scope='conv_2x2')

        # Paddings + second convolution.
        paddings = [[0, 0], [rate, rate], [rate, rate], [0, 0]]
        output = tf.pad(output, paddings, mode='CONSTANT')
        output = slim.conv2d(output, num_outputs, [2, 2], rate=rate,
                             padding='VALID',
                             activation_fn=activation_fn,
                             # normalizer_fn=normalizer_fn,
                             # normalizer_params=normalizer_params,
                             normalizer_fn=None,
                             normalizer_params=None,
                             weights_initializer=initializers.xavier_initializer_conv2d(),
                             weights_regularizer=weights_regularizer,
                             # biases_initializer=None,
                             biases_initializer=init_ops.zeros_initializer,
                             biases_regularizer=biases_regularizer,
                             scope='conv_concat')
        return output


# ==============================================================================
# Tiny Conv2d Complex!
# ==============================================================================
@add_arg_scope
def conv2d_tiny_complex(inputs,
                num_outputs,
                rate=1,
                padding='SAME',
                data_format=None,
                activation_fn=nn.relu,
                normalizer_fn=None,
                normalizer_params=None,
                weights_initializer=initializers.xavier_initializer_conv2d(),
                weights_regularizer=None,
                biases_initializer=init_ops.zeros_initializer,
                biases_regularizer=None,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=True,
                scope=None,):
    """Tiny Convolution 2d.
    """
    with variable_scope.variable_scope(scope, 'Conv', [inputs],
                                       reuse=reuse) as sc:
        inputs = ops.convert_to_tensor(inputs)
        dtype = inputs.dtype.base_dtype
        input_rank = inputs.get_shape().ndims
        if input_rank is None:
            raise ValueError('Rank of inputs must be known')
        if input_rank < 3 or input_rank > 5:
            raise ValueError('Rank of inputs is %d, which is not >= 3 and <= 5' %
                             input_rank)
        conv_dims = input_rank - 2

        # First 2x2 convolution.
        num_outputs_inter = num_outputs // 4
        out_list = []
        paddings = [[[0, 0], [0, rate], [0, rate], [0, 0]],
                    [[0, 0], [0, rate], [rate, 0], [0, 0]],
                    [[0, 0], [rate, 0], [0, rate], [0, 0]],
                    [[0, 0], [rate, 0], [rate, 0], [0, 0]]]
        for i in range(4):
            output = slim.conv2d(inputs, num_outputs_inter, [2, 2], rate=rate,
                                 padding='VALID',
                                 activation_fn=activation_fn,
                                 normalizer_fn=normalizer_fn,
                                 normalizer_params=normalizer_params,
                                 weights_initializer=weights_initializer,
                                 weights_regularizer=weights_regularizer,
                                 biases_initializer=biases_regularizer,
                                 biases_regularizer=biases_regularizer,
                                 scope='conv_2x2_%i' % i)
            out_list.append(tf.pad(output, paddings[i], mode='CONSTANT'))
            print(out_list[-1].get_shape())
            # out_list.append(output)

        # Concatening outputs.
        output = tf.concat(input_rank-1, out_list)
        return output







