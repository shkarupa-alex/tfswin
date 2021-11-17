import numpy as np
import tensorflow as tf
from keras import keras_parameterized, testing_utils
from tfswin.winatt import WindowAttention


@keras_parameterized.run_all_keras_modes
class TestWindowAttention(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            WindowAttention,
            kwargs={'window_size': 7, 'num_heads': 3, 'qkv_bias': True, 'qk_scale': None, 'attn_mask': False,
                    'attn_drop': 0., 'proj_drop': 0.},
            input_shape=[2, 49, 96],
            input_dtype='float32',
            expected_output_shape=[None, 49, 96],
            expected_output_dtype='float32'
        )

    def test_masked(self):
        kwargs = {'window_size': 7, 'num_heads': 24, 'qkv_bias': True, 'qk_scale': None, 'attn_mask': True,
                  'attn_drop': 0., 'proj_drop': 0.}
        layer = WindowAttention(**kwargs)
        inputs = [np.zeros((2, 49, 96), 'float32'), np.zeros((1, 49, 49), 'float32')]

        outputs = layer(inputs)
        outputs = self.evaluate(outputs)
        self.assertTupleEqual(outputs.shape, (2, 49, 96))
        self.assertEqual(outputs.dtype, inputs[0].dtype)


if __name__ == '__main__':
    tf.test.main()
