import numpy as np
from keras.src import ops, testing
from tfswin.window import window_partition, window_reverse, window_partition_fused, window_reverse_fused


class TestWindowAttention(testing.TestCase):
    def test_partition_fused(self):
        batch, height, width = 2, 24, 48
        qkv_mult, num_heads, channels = 3, 4, 5
        window = 8
        inputs = np.random.uniform(size=(batch, height, width, qkv_mult * num_heads * channels)).astype('float32')

        expected = window_partition(inputs, height, width, window)
        expected = ops.reshape(expected, [-1, window ** 2, 3, num_heads, channels])
        expected = ops.transpose(expected, [2, 0, 3, 1, 4])

        result = window_partition_fused(inputs, height, width, window, num_heads, qkv_mult=qkv_mult)

        self.assertAllClose(expected, result)

    def test_reverse_fused(self):
        batch, height, width = 2, 24, 48
        qkv_mult, num_heads, channels = 3, 4, 5
        window = 8

        inputs = np.random.uniform(
            size=(batch * height * width // window ** 2, num_heads, window ** 2, channels)).astype('float32')

        expected = ops.transpose(inputs, [0, 2, 1, 3])
        expected = ops.reshape(expected, [-1, window ** 2, channels * num_heads])
        expected = window_reverse(expected, height, width, window)

        result = window_reverse_fused(inputs, height, width, window, num_heads)

        self.assertAllClose(expected, result)
