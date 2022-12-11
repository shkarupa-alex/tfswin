import numpy as np
import tensorflow as tf
from keras import layers
from keras.testing_infra import test_combinations
from keras.saving.object_registration import register_keras_serializable
from tfswin.swin import SwinBlock
from testing_utils import layer_multi_io_test


@register_keras_serializable('TFSwin')
class SwinBlockSqueeze(SwinBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4), layers.InputSpec(ndim=1, dtype='int32'), layers.InputSpec(ndim=1, dtype='int32'),
            layers.InputSpec(ndim=2, dtype='int32'), layers.InputSpec(ndim=5)]

    def call(self, inputs, **kwargs):
        inputs, shift_size, window_size, relative_index, attention_mask = inputs

        return super().call([inputs, shift_size[0], window_size[0], relative_index[0], attention_mask], **kwargs)


@test_combinations.run_all_keras_modes
class TestSwinBlock(test_combinations.TestCase):
    def test_layer(self):
        inputs = 10 * np.random.random((1, 7, 7, 96)) - 0.5
        shift3 = np.array([3], 'int32')
        shift5 = np.array([5], 'int32')
        window = np.array([7], 'int32')
        index = np.zeros([1, 7 ** 4], 'int32')
        masks = 10 * np.random.random((1, 1, 1, 49, 49)) - 0.5

        layer_multi_io_test(
            SwinBlockSqueeze,
            kwargs={'num_heads': 24, 'mlp_ratio': 4., 'qkv_bias': True, 'qk_scale': None, 'drop': 0., 'attn_drop': 0.,
                    'path_drop': 0.20000000298023224, 'window_pretrain': 7, 'swin_v2': False},
            input_datas=[inputs, shift3, window, index, masks],
            input_dtypes=['float32', 'int32', 'int32', 'int32', 'float32'],
            expected_output_shapes=[(None, 7, 7, 96)],
            expected_output_dtypes=['float32']
        )
        layer_multi_io_test(
            SwinBlockSqueeze,
            kwargs={'num_heads': 24, 'mlp_ratio': 4., 'qkv_bias': True, 'qk_scale': None, 'drop': 0., 'attn_drop': 0.,
                    'path_drop': 0.20000000298023224, 'window_pretrain': 7, 'swin_v2': False},
            input_datas=[inputs, shift5, window, index, masks],
            input_dtypes=['float32', 'int32', 'int32', 'int32', 'float32'],
            expected_output_shapes=[(None, 7, 7, 96)],
            expected_output_dtypes=['float32']
        )

        layer_multi_io_test(
            SwinBlockSqueeze,
            kwargs={'num_heads': 24, 'mlp_ratio': 4., 'qkv_bias': True, 'qk_scale': None, 'drop': 0., 'attn_drop': 0.,
                    'path_drop': 0.20000000298023224, 'window_pretrain': 0, 'swin_v2': True},
            input_datas=[inputs, shift3, window, index, masks],
            input_dtypes=['float32', 'int32', 'int32', 'int32', 'float32'],
            expected_output_shapes=[(None, 7, 7, 96)],
            expected_output_dtypes=['float32']
        )
        layer_multi_io_test(
            SwinBlockSqueeze,
            kwargs={'num_heads': 24, 'mlp_ratio': 4., 'qkv_bias': True, 'qk_scale': None, 'drop': 0., 'attn_drop': 0.,
                    'path_drop': 0.20000000298023224, 'window_pretrain': 4, 'swin_v2': True},
            input_datas=[inputs, shift5, window, index, masks],
            input_dtypes=['float32', 'int32', 'int32', 'int32', 'float32'],
            expected_output_shapes=[(None, 7, 7, 96)],
            expected_output_dtypes=['float32']
        )


if __name__ == '__main__':
    tf.test.main()
