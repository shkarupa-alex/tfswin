import tensorflow as tf
from keras import keras_parameterized, testing_utils
from tfswin.swin import SwinBlock


@keras_parameterized.run_all_keras_modes
class TestSwinBlock(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            SwinBlock,
            kwargs={'num_heads': 24, 'window_size': 7, 'shift_size': 3, 'mlp_ratio': 4., 'qkv_bias': True,
                    'qk_scale': None, 'drop': 0., 'attn_drop': 0., 'path_drop': 0.20000000298023224},
            input_shape=[2, 49, 768],
            input_dtype='float32',
            expected_output_shape=[None, 49, 768],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
