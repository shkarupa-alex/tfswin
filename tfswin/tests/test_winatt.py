import numpy as np
import os
import tensorflow as tf
from keras import keras_parameterized, testing_utils
from ..winatt import WindowAttention


@keras_parameterized.run_all_keras_modes
class TestWindowAttention(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            WindowAttention,
            kwargs={'window_size': 7, 'num_heads': 3, 'qkv_bias': True, 'qk_scale': None,
                    'attn_drop': 0., 'proj_drop': 0.},
            input_shape=[2, 49, 96],
            input_dtype='float32',
            expected_output_shape=[None, 49, 96],
            expected_output_dtype='float32'
        )

    def test_value(self):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        inputs = np.load(f'{data_dir}/_winatt_input.npy')
        targets = np.load(f'{data_dir}/_winatt_output.npy')
        layer = WindowAttention(7, 24, True, None, 0., 0.)
        layer(inputs)  # build
        layer.set_weights([
            # layer.get_weights()[0],
            np.load(f'{data_dir}/_winatt_rel_bias.npy'),
            np.load(f'{data_dir}/_winatt_qkv_weight.npy').T,
            np.load(f'{data_dir}/_winatt_qkv_bias.npy').T,
            np.load(f'{data_dir}/_winatt_proj_weight.npy').T,
            np.load(f'{data_dir}/_winatt_proj_bias.npy').T,
            # layer.get_weights()[-1]
            # np.load(f'{data_dir}/_winatt_rel_index.npy'),
        ])
        outputs = self.evaluate(layer(inputs))
        self.assertLess(np.abs(targets - outputs).max(), 2.87e-6)


if __name__ == '__main__':
    tf.test.main()
