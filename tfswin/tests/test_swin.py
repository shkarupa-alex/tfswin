import tensorflow as tf
from keras import keras_parameterized
from tfswin.swin import SwinBlock
from testing_utils import layer_multi_io_test


@keras_parameterized.run_all_keras_modes
class TestSwinBlock(keras_parameterized.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            SwinBlock,
            kwargs={'num_heads': 24, 'window_size': 7, 'shift_size': 3, 'mlp_ratio': 4., 'qkv_bias': True,
                    'qk_scale': None, 'drop': 0., 'attn_drop': 0., 'path_drop': 0.20000000298023224},
            input_shapes=[(1, 7, 7, 768), (1, 1, 1, 49, 49)],
            input_dtypes=['float32'] * 2,
            expected_output_shapes=[(None, 7, 7, 768)],
            expected_output_dtypes=['float32']
        )
        layer_multi_io_test(
            SwinBlock,
            kwargs={'num_heads': 24, 'window_size': 7, 'shift_size': 5, 'mlp_ratio': 4., 'qkv_bias': True,
                    'qk_scale': None, 'drop': 0., 'attn_drop': 0., 'path_drop': 0.20000000298023224},
            input_shapes=[(1, 7, 7, 768), (1, 1, 1, 49, 49)],
            input_dtypes=['float32'] * 2,
            expected_output_shapes=[(None, 7, 7, 768)],
            expected_output_dtypes=['float32']
        )


if __name__ == '__main__':
    tf.test.main()
