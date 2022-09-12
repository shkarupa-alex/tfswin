import tensorflow as tf
from keras.applications import imagenet_utils
from keras.mixed_precision import global_policy


def preprocess_input(inputs, dtype=None):
    if dtype is None:
        dtype = global_policy().compute_dtype

    inputs = tf.cast(inputs, dtype)
    outputs = imagenet_utils.preprocess_input(inputs, data_format='channels_last', mode='torch')

    return outputs
