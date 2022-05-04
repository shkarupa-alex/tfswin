import tensorflow as tf


def window_partition(inputs, height, width, window_size, dtype=None, name=None):
    with tf.name_scope(name or 'window_partition'):
        inputs = tf.convert_to_tensor(inputs, dtype)

        if 4 != inputs.shape.rank:
            raise ValueError('Expecting inputs rank to be 4.')

        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError('Channel dimensions of the inputs should be defined. Found `None`.')

        windows_height = height // window_size
        windows_width = width // window_size

        outputs = tf.reshape(inputs, [-1, windows_height, window_size, windows_width, window_size, channels])
        outputs = tf.transpose(outputs, [0, 1, 3, 2, 4, 5])
        outputs = tf.reshape(outputs, [-1, window_size ** 2, channels])

        return outputs


def window_reverse(inputs, height, width, window_size, dtype=None, name=None):
    with tf.name_scope(name or 'window_reverse'):
        inputs = tf.convert_to_tensor(inputs, dtype)

        if 3 != inputs.shape.rank:
            raise ValueError('Expecting inputs rank to be 3.')

        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError('Channel dimensions of the inputs should be defined. Found `None`.')

        windows_height = height // window_size
        windows_width = width // window_size

        outputs = tf.reshape(inputs, [-1, windows_height, windows_width, window_size, window_size, channels])
        outputs = tf.transpose(outputs, [0, 1, 3, 2, 4, 5])
        outputs = tf.reshape(outputs, [-1, height, width, channels])

        return outputs
