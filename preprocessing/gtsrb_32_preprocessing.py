# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides utilities to preprocess images in CIFAR-10.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import random

_PADDING = 12

slim = tf.contrib.slim


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
    """Distort the color of a Tensor image.

    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.

    Args:
        image: 3-D Tensor containing single image in [0, 1].
        color_ordering: Python int, a type of distortion (valid values: 0-3).
        fast_mode: Avoids slower ops (random_hue and random_contrast)
        scope: Optional scope for name_scope.
    Returns:
        3-D Tensor color-distorted image on range [0, 1]
    Raises:
        ValueError: if color_ordering not in [0, 3]
    """
    with tf.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                raise ValueError('color_ordering must be in [0, 3]')

        # The random_* ops do not necessarily clamp.
        return tf.clip_by_value(image, 0.0, 1.0)


def preprocess_for_train(image,
                         output_height,
                         output_width,
                         padding=_PADDING):
    """Preprocesses the given image for training.
    Note that the actual resizing scale is sampled from
    [`resize_size_min`, `resize_size_max`].

    Args:
      image: A `Tensor` representing an image of arbitrary size.
      output_height: The height of the image after preprocessing.
      output_width: The width of the image after preprocessing.
      padding: The amound of padding before and after each dimension of the image.

    Returns:
      A preprocessed image.
    """
    tf.image_summary('image', tf.expand_dims(image, 0))

    # Transform the image to float if necessary (and rescale to 0-1)
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # image = tf.to_float(image)

    # Randomly flip the image horizontally.
    # Not a good idea for traffic sign!!!
    # distorted_image = tf.image.random_flip_left_right(distorted_image)
    # Random contrast and brightness.
    # distorted_image = tf.image.random_brightness(distorted_image,
    #                                              max_delta=63)
    # distorted_image = tf.image.random_contrast(distorted_image,
    #                                            lower=0.2, upper=1.8)
    # Translate to [-1, 1] interval.
    # distorted_image = tf.sub(distorted_image, 0.5)
    # distorted_image = tf.mul(distorted_image, 2.0)

    # Random color distortion.
    ordering = random.randint(0, 3)
    print(ordering)
    distorted_image = distort_color(image,
                                    color_ordering=ordering,
                                    fast_mode=False)
    tf.image_summary('distorted_image', tf.expand_dims(distorted_image, 0))

    # Whitened image.
    whitened_image = tf.image.per_image_whitening(distorted_image)
    tf.image_summary('whitened_image', tf.expand_dims(whitened_image, 0))

    # Randomly crop a [height, width] section of the image.
    if padding > 0:
        whitened_image = tf.pad(whitened_image,
                                [[padding, padding], [padding, padding], [0, 0]],
                                mode='CONSTANT')
    whitened_image = tf.random_crop(whitened_image,
                                    [output_height, output_width, 3])
    return whitened_image


def preprocess_for_eval(image, output_height, output_width):
    """Preprocesses the given image for evaluation.

    Args:
      image: A `Tensor` representing an image of arbitrary size.
      output_height: The height of the image after preprocessing.
      output_width: The width of the image after preprocessing.

    Returns:
      A preprocessed image.
    """
    tf.image_summary('image', tf.expand_dims(image, 0))
    # Transform the image to float if necessary (and rescale to 0-1)
    # if image.dtype != tf.float32:
    #     image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.to_float(image)

    # Resize and crop if needed.
    resized_image = tf.image.resize_image_with_crop_or_pad(image,
                                                           output_height,
                                                           output_width)
    tf.image_summary('resized_image', tf.expand_dims(resized_image, 0))

    # Translate to [-1, 1] interval.
    # resized_image = tf.sub(resized_image, 0.5)
    # resized_image = tf.mul(resized_image, 2.0)
    return tf.image.per_image_whitening(resized_image)


def preprocess_image(image, output_height, output_width, is_training=False):
    """Preprocesses the given image.

    Args:
      image: A `Tensor` representing an image of arbitrary size.
      output_height: The height of the image after preprocessing.
      output_width: The width of the image after preprocessing.
      is_training: `True` if we're preprocessing the image for training and
        `False` otherwise.

    Returns:
      A preprocessed image.
    """
    # print(image.dtype)

    if is_training:
        return preprocess_for_train(image, output_height, output_width)
    else:
        return preprocess_for_eval(image, output_height, output_width)
