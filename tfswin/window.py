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


def window_partition_fused(inputs, height, width, window_size, num_heads, qkv_mult=3, dtype=None, name=None):
    with tf.name_scope(name or 'window_partition'):
        inputs = tf.convert_to_tensor(inputs, dtype)

        if 4 != inputs.shape.rank:
            raise ValueError('Expecting inputs rank to be 4.')

        full_channels = inputs.shape[-1]
        if full_channels is None:
            raise ValueError('Channel dimensions of the inputs should be defined. Found `None`.')

        windows_height = height // window_size
        windows_width = width // window_size
        head_channels = full_channels // (qkv_mult * num_heads)

        outputs = tf.reshape(
            inputs, [-1, windows_height, window_size, windows_width, window_size, qkv_mult, num_heads, head_channels])
        outputs = tf.transpose(outputs, [5, 0, 1, 3, 6, 2, 4, 7])
        outputs = tf.reshape(outputs, [qkv_mult, -1, num_heads, window_size ** 2, head_channels])

        return outputs


def window_reverse_fused(inputs, height, width, window_size, num_heads, dtype=None, name=None):
    with tf.name_scope(name or 'window_reverse'):
        inputs = tf.convert_to_tensor(inputs, dtype)

        if 4 != inputs.shape.rank:
            raise ValueError('Expecting inputs rank to be 4.')

        head_channels = inputs.shape[-1]
        if head_channels is None:
            raise ValueError('Channel dimensions of the inputs should be defined. Found `None`.')

        windows_height = height // window_size
        windows_width = width // window_size
        full_channels = head_channels * num_heads

        outputs = tf.reshape(
            inputs, [-1, windows_height, windows_width, num_heads, window_size, window_size, head_channels])
        outputs = tf.transpose(outputs, [0, 1, 4, 2, 5, 3, 6])
        outputs = tf.reshape(outputs, [-1, height, width, full_channels])

        return outputs
