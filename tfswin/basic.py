import numpy as np
from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from tfswin.merge import PatchMerging
from tfswin.swin import SwinBlock


@register_keras_serializable(package='TFSwin')
class BasicLayer(layers.Layer):
    def __init__(self, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 path_drop=0., downsample=False, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=3)

        self.depth = depth
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop = drop
        self.attn_drop = attn_drop
        self.path_drop = path_drop
        self.downsample = downsample

    @shape_type_conversion
    def build(self, input_shape):
        path_drop = self.path_drop
        if not isinstance(self.path_drop, (list, tuple)):
            path_drop = [self.path_drop] * self.depth

        shift_size = np.zeros(self.depth, 'int32')
        shift_size[1::2] = self.window_size // 2

        # noinspection PyAttributeOutsideInit
        self.blocks = [
            SwinBlock(num_heads=self.num_heads, window_size=self.window_size, shift_size=shift_size[i],
                                 mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                                 drop=self.drop, attn_drop=self.attn_drop, path_drop=path_drop[i], name=f'blocks.{i}')
            for i in range(self.depth)]

        if self.downsample:
            # noinspection PyAttributeOutsideInit
            self.down = PatchMerging(name='downsample')

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        outputs = inputs
        for b in self.blocks:
            outputs = b(outputs)

        if self.downsample:
            outputs = self.down(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        if self.downsample:
            return self.down.compute_output_shape(input_shape)

        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'depth': self.depth,
            'num_heads': self.num_heads,
            'window_size': self.window_size,
            'mlp_ratio': self.mlp_ratio,
            'qkv_bias': self.qkv_bias,
            'qk_scale': self.qk_scale,
            'drop': self.drop,
            'attn_drop': self.attn_drop,
            'path_drop': self.path_drop,
            'downsample': self.downsample,
        })

        return config
