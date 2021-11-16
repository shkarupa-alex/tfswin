import numpy as np
import os
import tensorflow as tf
from keras import keras_parameterized, testing_utils
from ..basic import BasicLayer


@keras_parameterized.run_all_keras_modes
class TestBasicLayer(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            BasicLayer,
            kwargs={'dim': 96, 'input_resolution': (56, 56), 'depth': 2, 'num_heads': 3, 'window_size': 7,
                    'mlp_ratio': 4., 'qkv_bias': True, 'qk_scale': None, 'drop': 0., 'attn_drop': 0.,
                    'drop_path_prob': [0.0, 0.0181818176060915], 'downsample': True},
            input_shape=[2, 3136, 96],
            input_dtype='float32',
            expected_output_shape=[None, 784, 192],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
