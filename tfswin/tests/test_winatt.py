import numpy as np
import tensorflow as tf
from keras import layers
from keras.testing_infra import test_combinations
from keras.saving.object_registration import register_keras_serializable
from tfswin.winatt import WindowAttention
from testing_utils import layer_multi_io_test


@register_keras_serializable('TFSwin')
class WindowAttentionSqueeze(WindowAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4), layers.InputSpec(ndim=1, dtype='int32'), layers.InputSpec(ndim=2, dtype='int32'),
            layers.InputSpec(ndim=5)]

    def call(self, inputs, **kwargs):
        inputs, window_size, relative_index, attention_mask = inputs

        return super().call([inputs, window_size[0], relative_index[0], attention_mask], **kwargs)


@test_combinations.run_all_keras_modes
class TestWindowAttention(test_combinations.TestCase):
    def test_layer(self):
        inputs = 10 * np.random.random((1, 7, 7, 96)) - 0.5
        window = np.array([7], 'int32')
        index = np.zeros([1, 7 ** 4], 'int32')
        masks = 10 * np.random.random((1, 1, 1, 49, 49)) - 0.5

        layer_multi_io_test(
            WindowAttentionSqueeze,
            kwargs={'num_heads': 3, 'qkv_bias': True, 'qk_scale': None, 'attn_drop': 0., 'proj_drop': 0.,
                    'window_pretrain': 7, 'swin_v2': False},
            input_datas=[inputs, window, index, masks],
            input_dtypes=['float32', 'int32', 'int32', 'float32'],
            expected_output_shapes=[(None, None, None, 96)],
            expected_output_dtypes=['float32']
        )
        layer_multi_io_test(
            WindowAttentionSqueeze,
            kwargs={'num_heads': 3, 'qkv_bias': True, 'qk_scale': None, 'attn_drop': 0., 'proj_drop': 0.,
                    'window_pretrain': 0, 'swin_v2': False},
            input_datas=[inputs, window, index, masks],
            input_dtypes=['float32', 'int32', 'int32', 'float32'],
            expected_output_shapes=[(None, None, None, 96)],
            expected_output_dtypes=['float32']
        )

        layer_multi_io_test(
            WindowAttentionSqueeze,
            kwargs={'num_heads': 3, 'qkv_bias': True, 'qk_scale': None, 'attn_drop': 0., 'proj_drop': 0.,
                    'window_pretrain': 0, 'swin_v2': True},
            input_datas=[inputs, window, index, masks],
            input_dtypes=['float32', 'int32', 'int32', 'float32'],
            expected_output_shapes=[(None, None, None, 96)],
            expected_output_dtypes=['float32']
        )
        layer_multi_io_test(
            WindowAttentionSqueeze,
            kwargs={'num_heads': 3, 'qkv_bias': True, 'qk_scale': None, 'attn_drop': 0., 'proj_drop': 0.,
                    'window_pretrain': 8, 'swin_v2': True},
            input_datas=[inputs, window, index, masks],
            input_dtypes=['float32', 'int32', 'int32', 'float32'],
            expected_output_shapes=[(None, None, None, 96)],
            expected_output_dtypes=['float32']
        )


if __name__ == '__main__':
    tf.test.main()
