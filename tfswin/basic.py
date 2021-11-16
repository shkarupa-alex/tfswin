import tensorflow as tf
from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from .merge import PatchMerging
from .swin import SwinTransformerBlock


@register_keras_serializable(package='TFSwin')
class BasicLayer(layers.Layer):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path_prob=0., downsample=False, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop = drop
        self.attn_drop = attn_drop
        self.drop_path_prob = drop_path_prob
        self.downsample = downsample

        # build blocks
        self.blocks = [
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path_prob=drop_path_prob[i] if isinstance(drop_path_prob, list) else drop_path_prob,
                name=f'blocks.{i}'
            ) for i in range(depth)]

        self.down = None
        if downsample:
            self.down = PatchMerging(name='downsample')

    def call(self, x):
        for b in self.blocks:
            x = b(x)

        if self.down is not None:
            x = self.down(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'input_resolution': self.input_resolution,
            'depth': self.depth,
            'num_heads': self.num_heads,
            'window_size': self.window_size,
            'mlp_ratio': self.mlp_ratio,
            'qkv_bias': self.qkv_bias,
            'qk_scale': self.qk_scale,
            'drop': self.drop,
            'attn_drop': self.attn_drop,
            'drop_path_prob': self.drop_path_prob,
            'downsample': self.downsample,
        })

        return config
