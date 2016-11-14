# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a variant of the Xception model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


# ==============================================================================
# Xception model.
# ==============================================================================
def xception(images, num_classes=43, is_training=False,
             dropout_keep_prob=0.5,
             prediction_fn=slim.softmax,
             scope='Xception'):
    """Creates a model based on Xception.
    Returns:
    logits: the pre-softmax activations, a tensor of size
        [batch_size, `num_classes`]
    end_points: a dictionary from components of the network to the corresponding
        activation.
    """
    end_points = {}

    with tf.variable_scope(scope, 'Xception', [images, num_classes]):
        # First two layers: normal convolutions, with VALID padding.
        net = slim.conv2d(images, 32, [3, 3], padding='VALID',
                          scope='conv1')
        end_points['conv1'] = net
        net = slim.conv2d(net, 64, [3, 3], padding='VALID',
                          scope='conv2')
        end_points['conv2'] = net
        net = slim.max_pool2d(net, [3, 3], 1, scope='pool1', padding='SAME')

        net = slim.conv2d(net, 128, [3, 3], padding='VALID',
                          scope='conv3')
        end_points['conv3'] = net
        net = slim.conv2d(net, 256, [3, 3], padding='VALID',
                          scope='conv4')
        end_points['conv4'] = net

        # # Separable convolutions for the next layers!
        # net = slim.separable_convolution2d(net, 128, [3, 3], 1,
        #                                    scope='conv_sep1')
        # end_points['conv_sep1'] = net
        # net = slim.separable_convolution2d(net, 256, [3, 3], 1,
        #                                    scope='conv_sep2')
        # end_points['conv_sep2'] = net
        # net = slim.max_pool2d(net, [3, 3], 1, scope='pool2', padding='SAME')

        # # Second stack
        # net = slim.separable_convolution2d(net, 512, [3, 3], 1,
        #                                    scope='conv_sep3')
        # end_points['conv_sep3'] = net
        # net = slim.separable_convolution2d(net, 512, [3, 3], 1,
        #                                    scope='conv_sep4')
        # end_points['conv_sep4'] = net
        # net = slim.max_pool2d(net, [3, 3], 1, scope='pool3', padding='SAME')

        # Fully connected convolution + Logits.
        net = slim.conv2d(net, 512, [1, 1],
                          normalizer_fn=None,
                          scope='conv_fc')
        end_points['conv_fc'] = net
        net = slim.dropout(net, dropout_keep_prob,
                           is_training=is_training,
                           scope='dropout1')
        net = slim.conv2d(net, num_classes+1, [1, 1],
                          biases_initializer=tf.zeros_initializer,
                          # weights_initializer=trunc_normal(1 / 512.0),
                          weights_regularizer=None,
                          activation_fn=None,
                          scope='conv_logits')
        end_points['conv_logits'] = net
        end_points['PredictionsFull'] = tf.nn.softmax(net)

        # Global average pooling.
        logits = tf.reduce_mean(net, [1, 2], name='pool7')

        end_points['Logits'] = logits
        end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return logits, end_points
xception.default_image_size = 32


def xception_arg_scope(weight_decay=0.004):
    """Defines the default argument scope.

    Args:
        weight_decay: The weight decay to use for regularizing the model.
    Returns:
        An `arg_scope` to use for the inception v3 model.
    """
    batch_norm_params = {
        # 'center': False,
        'scale': False,
        # Decay for the moving averages.
        'decay': 0.9997,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.01,
        # collection containing update_ops.
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }
    with slim.arg_scope(
            [slim.conv2d, slim.separable_convolution2d],
            weights_initializer=tf.uniform_unit_scaling_initializer(factor=1.43),
            weights_regularizer=slim.l2_regularizer(weight_decay),
            # weights_regularizer=None,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params,
            activation_fn=tf.nn.relu) as sc:
        return sc
