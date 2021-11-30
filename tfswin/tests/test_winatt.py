import numpy as np
import tensorflow as tf
from keras import keras_parameterized, layers
from keras.utils.generic_utils import register_keras_serializable
from tfswin.winatt import WindowAttention
from testing_utils import layer_multi_io_test


@register_keras_serializable('TFSwin')
class WindowAttentionSqueeze(WindowAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_spec = self.input_spec[:-1] + [layers.InputSpec(ndim=1, dtype='bool')]

    def build(self, input_shape):
        super().build(input_shape)
        self.input_spec = self.input_spec[:-1] + [layers.InputSpec(ndim=1, dtype='bool')]

    def call(self, inputs, **kwargs):
        inputs, mask, with_mask = inputs

        return super().call([inputs, mask, tf.squeeze(with_mask, axis=0)], **kwargs)


@keras_parameterized.run_all_keras_modes
class TestWindowAttention(keras_parameterized.TestCase):
    def test_layer(self):
        inputs = 10 * np.random.random((1, 49, 96)) - 0.5
        masks = 10 * np.random.random((1, 1, 1, 49, 49)) - 0.5

        layer_multi_io_test(
            WindowAttentionSqueeze,
            kwargs={'window_size': 7, 'num_heads': 3, 'qkv_bias': True, 'qk_scale': None, 'attn_drop': 0.,
                    'proj_drop': 0.},
            input_datas=[inputs, masks, np.array([False])],
            input_dtypes=['float32', 'float32', 'bool'],
            expected_output_shapes=[(None, 49, 96)],
            expected_output_dtypes=['float32']
        )
        layer_multi_io_test(
            WindowAttentionSqueeze,
            kwargs={'window_size': 7, 'num_heads': 3, 'qkv_bias': True, 'qk_scale': None, 'attn_drop': 0.,
                    'proj_drop': 0.},
            input_datas=[inputs, masks, np.array([True])],
            input_dtypes=['float32', 'float32', 'bool'],
            expected_output_shapes=[(None, 49, 96)],
            expected_output_dtypes=['float32']
        )


if __name__ == '__main__':
    tf.test.main()
