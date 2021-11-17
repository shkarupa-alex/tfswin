import numpy as np
import os
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

    def test_value(self):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        inputs = np.load(f'{data_dir}/_swin_input.npy')
        targets = np.load(f'{data_dir}/_swin_output.npy')
        layer = SwinBlock(24, 7, 3, 4.0, True, None, 0.0, 0.0, 0.20000000298023224)
        layer(inputs)  # build
        layer.set_weights([
            np.load(f'{data_dir}/_swin_norm1_weight.npy').T,
            np.load(f'{data_dir}/_swin_norm1_bias.npy').T,

            # layer.get_weights()[0],
            np.load(f'{data_dir}/_swin_winatt_rel_bias.npy'),
            np.load(f'{data_dir}/_swin_winatt_qkv_weight.npy').T,
            np.load(f'{data_dir}/_swin_winatt_qkv_bias.npy').T,
            np.load(f'{data_dir}/_swin_winatt_proj_weight.npy').T,
            np.load(f'{data_dir}/_swin_winatt_proj_bias.npy').T,

            np.load(f'{data_dir}/_swin_norm2_weight.npy').T,
            np.load(f'{data_dir}/_swin_norm2_bias.npy').T,

            np.load(f'{data_dir}/_swin_mlp_fc1_weight.npy').T,
            np.load(f'{data_dir}/_swin_mlp_fc1_bias.npy').T,
            np.load(f'{data_dir}/_swin_mlp_fc2_weight.npy').T,
            np.load(f'{data_dir}/_swin_mlp_fc2_bias.npy').T,

            # layer.get_weights()[-1]
            # np.load(f'{data_dir}/_swin_winatt_rel_index.npy'),
        ])
        outputs = self.evaluate(layer(inputs))
        self.assertLess(np.abs(targets - outputs).max(), 4.58e-5)


if __name__ == '__main__':
    tf.test.main()
