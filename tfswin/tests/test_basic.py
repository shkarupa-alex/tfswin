import tensorflow as tf
from keras.testing_infra import test_combinations, test_utils
from tfswin.basic import BasicLayer


@test_combinations.run_all_keras_modes
class TestBasicLayer(test_combinations.TestCase):
    def test_layer(self):
        test_utils.layer_test(
            BasicLayer,
            kwargs={'depth': 2, 'num_heads': 3, 'window_size': 7, 'mlp_ratio': 4., 'qkv_bias': True, 'qk_scale': None,
                    'drop': 0., 'attn_drop': 0., 'path_drop': [0.0, 0.0181818176060915], 'window_pretrain': 7,
                    'swin_v2': False},
            input_shape=[2, 56, 56, 96],
            input_dtype='float32',
            expected_output_shape=[None, 56, 56, 96],
            expected_output_dtype='float32'
        )

        test_utils.layer_test(
            BasicLayer,
            kwargs={'depth': 2, 'num_heads': 3, 'window_size': 7, 'mlp_ratio': 4., 'qkv_bias': True, 'qk_scale': None,
                    'drop': 0., 'attn_drop': 0., 'path_drop': [0.0, 0.0181818176060915], 'window_pretrain': 4,
                    'swin_v2': True},
            input_shape=[2, 56, 56, 96],
            input_dtype='float32',
            expected_output_shape=[None, 56, 56, 96],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
