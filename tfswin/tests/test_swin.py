import numpy as np
from keras.src import layers, testing
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable
from tfswin.swin import SwinBlock


@register_keras_serializable('TFSwin')
class SwinBlockSqueeze(SwinBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_spec = [
            InputSpec(ndim=4), InputSpec(ndim=1, dtype='int32'), InputSpec(ndim=1, dtype='int32'),
            InputSpec(ndim=2, dtype='int32'), InputSpec(ndim=5)]

    def call(self, inputs, **kwargs):
        inputs, shift_size, window_size, relative_index, attention_mask = inputs

        return super().call([inputs, shift_size[0], window_size[0], relative_index[0], attention_mask], **kwargs)


class TestSwinBlock(testing.TestCase):
    def test_layer(self):
        inputs = 10 * np.random.random((1, 7, 7, 96)) - 0.5
        shift3 = np.array([3], 'int32')
        shift5 = np.array([5], 'int32')
        window = np.array([7], 'int32')
        index = np.zeros([1, 7 ** 4], 'int32')
        masks = 10 * np.random.random((1, 1, 1, 49, 49)) - 0.5

        self.run_layer_test(
            SwinBlockSqueeze,
            init_kwargs={'num_heads': 24, 'mlp_ratio': 4., 'qkv_bias': True, 'qk_scale': None, 'drop': 0., 'attn_drop': 0.,
                    'path_drop': 0.20000000298023224, 'window_pretrain': 7, 'swin_v2': False},
            input_data=(inputs, shift3, window, index, masks),
            input_dtype=('float32', 'int32', 'int32', 'int32', 'float32'),
            expected_output_shape=(1, 7, 7, 96),
            expected_output_dtype='float32'
        )
        self.run_layer_test(
            SwinBlockSqueeze,
            init_kwargs={'num_heads': 24, 'mlp_ratio': 4., 'qkv_bias': True, 'qk_scale': None, 'drop': 0., 'attn_drop': 0.,
                    'path_drop': 0.20000000298023224, 'window_pretrain': 7, 'swin_v2': False},
            input_data=(inputs, shift5, window, index, masks),
            input_dtype=('float32', 'int32', 'int32', 'int32', 'float32'),
            expected_output_shape=(1, 7, 7, 96),
            expected_output_dtype='float32'
        )

        self.run_layer_test(
            SwinBlockSqueeze,
            init_kwargs={'num_heads': 24, 'mlp_ratio': 4., 'qkv_bias': True, 'qk_scale': None, 'drop': 0., 'attn_drop': 0.,
                    'path_drop': 0.20000000298023224, 'window_pretrain': 0, 'swin_v2': True},
            input_data=(inputs, shift3, window, index, masks),
            input_dtype=('float32', 'int32', 'int32', 'int32', 'float32'),
            expected_output_shape=(1, 7, 7, 96),
            expected_output_dtype='float32'
        )
        self.run_layer_test(
            SwinBlockSqueeze,
            init_kwargs={'num_heads': 24, 'mlp_ratio': 4., 'qkv_bias': True, 'qk_scale': None, 'drop': 0., 'attn_drop': 0.,
                    'path_drop': 0.20000000298023224, 'window_pretrain': 4, 'swin_v2': True},
            input_data=(inputs, shift5, window, index, masks),
            input_dtype=('float32', 'int32', 'int32', 'int32', 'float32'),
            expected_output_shape=(1, 7, 7, 96),
            expected_output_dtype='float32'
        )
