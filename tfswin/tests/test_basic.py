import tensorflow as tf
from keras.src import testing
from tfswin.basic import BasicLayer


class TestBasicLayer(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            BasicLayer,
            init_kwargs={'depth': 2, 'num_heads': 3, 'window_size': 7, 'mlp_ratio': 4., 'qkv_bias': True, 'qk_scale': None,
                    'drop': 0., 'attn_drop': 0., 'path_drop': [0.0, 0.0181818176060915], 'window_pretrain': 7,
                    'swin_v2': False},
            input_shape=(2, 56, 56, 96),
            input_dtype='float32',
            expected_output_shape=(2, 56, 56, 96),
            expected_output_dtype='float32'
        )
        # self.run_layer_test(
        #     BasicLayer,
        #     init_kwargs={'depth': 2, 'num_heads': 3, 'window_size': 7, 'mlp_ratio': 4., 'qkv_bias': True, 'qk_scale': None,
        #             'drop': 0., 'attn_drop': 0., 'path_drop': [0.0, 0.0181818176060915], 'window_pretrain': 4,
        #             'swin_v2': True},
        #     input_shape=(2, 56, 56, 96),
        #     input_dtype='float32',
        #     expected_output_shape=(2, 56, 56, 96),
        #     expected_output_dtype='float32'
        # )


if __name__ == '__main__':
    tf.test.main()
