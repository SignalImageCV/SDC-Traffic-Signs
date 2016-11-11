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
# Convolution 2D Padding
# ==============================================================================
def conv2d_pad(inputs,
               num_outputs,
               kernel_size,
               stride=1,
               padding='SAME',
               data_format=None,
               rate=1,
               activation_fn=nn.relu,
               normalizer_fn=None,
               normalizer_params=None,
               weights_initializer=initializers.xavier_initializer(),
               weights_regularizer=None,
               biases_initializer=init_ops.zeros_initializer,
               biases_regularizer=None,
               reuse=None,
               variables_collections=None,
               outputs_collections=None,
               trainable=True,
               scope=None):
    """Convolution 2D with correct padding.
    """
    if data_format not in [None, 'NWC', 'NCW', 'NHWC', 'NCHW', 'NDHWC']:
        raise ValueError('Invalid data_format: %r' % (data_format,))
    with variable_scope.variable_scope(scope, 'Conv', [inputs], reuse=reuse) as sc:
        inputs = ops.convert_to_tensor(inputs)
        dtype = inputs.dtype.base_dtype
        input_rank = inputs.get_shape().ndims
        if input_rank is None:
            raise ValueError('Rank of inputs must be known')
        if input_rank < 3 or input_rank > 5:
            raise ValueError('Rank of inputs is %d, which is not >= 3 and <= 5' %
                             input_rank)
        conv_dims = input_rank - 2
        kernel_size = utils.n_positive_integers(conv_dims, kernel_size)
        stride = utils.n_positive_integers(conv_dims, stride)
        rate = utils.n_positive_integers(conv_dims, rate)

        if data_format is None or data_format.endswith('C'):
            num_input_channels = inputs.get_shape()[input_rank - 1].value
        elif data_format.startswith('NC'):
            num_input_channels = inputs.get_shape()[1].value
        else:
            raise ValueError('Invalid data_format')

        if num_input_channels is None:
            raise ValueError('Number of in_channels must be known.')

        weights_shape = (
                list(kernel_size) + [num_input_channels, num_outputs])
        weights_collections = utils.get_variable_collections(variables_collections,
                                                             'weights')
        weights = variables.model_variable('weights',
                                           shape=weights_shape,
                                           dtype=dtype,
                                           initializer=weights_initializer,
                                           regularizer=weights_regularizer,
                                           collections=weights_collections,
                                           trainable=trainable)
        outputs = nn.convolution(input=inputs,
                                 filter=weights,
                                 dilation_rate=rate,
                                 strides=stride,
                                 padding=padding,
                                 data_format=data_format)
        if normalizer_fn is not None:
            normalizer_params = normalizer_params or {}
            outputs = normalizer_fn(outputs, **normalizer_params)
        else:
            if biases_initializer is not None:
                biases_collections = utils.get_variable_collections(
                        variables_collections, 'biases')
                biases = variables.model_variable('biases',
                                                  shape=[num_outputs],
                                                  dtype=dtype,
                                                  initializer=biases_initializer,
                                                  regularizer=biases_regularizer,
                                                  collections=biases_collections,
                                                  trainable=trainable)
                outputs = nn.bias_add(outputs, biases, data_format=data_format)
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return utils.collect_named_outputs(outputs_collections,
                                           sc.original_name_scope, outputs)


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







