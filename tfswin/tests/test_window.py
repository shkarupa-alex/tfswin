import numpy as np
import tensorflow as tf
from tf_keras.src.testing_infra import test_combinations
from tfswin.window import window_partition, window_reverse, window_partition_fused, window_reverse_fused


@test_combinations.run_all_keras_modes
class TestWindowAttention(test_combinations.TestCase):
    def test_partition_fused(self):
        batch, height, width = 2, 24, 48
        qkv_mult, num_heads, channels = 3, 4, 5
        window = 8
        inputs = np.random.uniform(size=(batch, height, width, qkv_mult * num_heads * channels)).astype('float32')

        expected = window_partition(inputs, height, width, window)
        expected = tf.reshape(expected, [-1, window ** 2, 3, num_heads, channels])
        expected = tf.transpose(expected, [2, 0, 3, 1, 4])
        expected = self.evaluate(expected)

        result = window_partition_fused(inputs, height, width, window, num_heads, qkv_mult=qkv_mult)
        result = self.evaluate(result)

        self.assertAllClose(expected, result)

    def test_reverse_fused(self):
        batch, height, width = 2, 24, 48
        qkv_mult, num_heads, channels = 3, 4, 5
        window = 8

        inputs = np.random.uniform(
            size=(batch * height * width // window ** 2, num_heads, window ** 2, channels)).astype('float32')

        expected = tf.transpose(inputs, perm=[0, 2, 1, 3])
        expected = tf.reshape(expected, [-1, window ** 2, channels * num_heads])
        expected = window_reverse(expected, height, width, window)
        expected = self.evaluate(expected)

        result = window_reverse_fused(inputs, height, width, window, num_heads)
        result = self.evaluate(result)

        self.assertAllClose(expected, result)


if __name__ == '__main__':
    tf.test.main()
