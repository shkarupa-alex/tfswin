import numpy as np
import tensorflow as tf
from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from tfswin.swin import SwinBlock
from tfswin.window import window_partition


@register_keras_serializable(package='TFSwin')
class BasicLayer(layers.Layer):
    def __init__(self, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., path_drop=0., **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.depth = depth
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop = drop
        self.attn_drop = attn_drop
        self.path_drop = path_drop

        self.shift_size = self.window_size // 2

    @shape_type_conversion
    def build(self, input_shape):
        path_drop = self.path_drop
        if not isinstance(self.path_drop, (list, tuple)):
            path_drop = [self.path_drop] * self.depth

        shift_size = np.zeros(self.depth, 'int32')
        shift_size[1::2] = self.shift_size

        # noinspection PyAttributeOutsideInit
        self.blocks = [
            SwinBlock(num_heads=self.num_heads, window_size=self.window_size, shift_size=shift_size[i],
                      mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale, drop=self.drop,
                      attn_drop=self.attn_drop, path_drop=path_drop[i], name=f'blocks.{i}')
            for i in range(self.depth)]

        super().build(input_shape)

    def attention_mask(self, inputs):
        height, width = tf.unstack(tf.shape(inputs)[1:3])
        padded_height = tf.cast(tf.math.ceil(height / self.window_size), 'int32') * self.window_size
        padded_width = tf.cast(tf.math.ceil(width / self.window_size), 'int32') * self.window_size

        last_repeats = [self.window_size - self.shift_size, self.shift_size]

        image_mask = np.arange(9, dtype='int32').reshape((3, 3))
        image_mask = tf.repeat(image_mask, [padded_height - self.window_size] + last_repeats, axis=1)
        image_mask = tf.repeat(image_mask, [padded_width - self.window_size] + last_repeats, axis=0)
        image_mask = image_mask[None, ..., None]

        mask_windows = window_partition(image_mask, padded_height, padded_width, self.window_size, 'int32')
        mask_windows = mask_windows[..., 0]

        attn_mask = mask_windows[:, None] - mask_windows[:, :, None]
        attn_mask = tf.where(attn_mask == 0, 0., -100.)
        attn_mask = tf.cast(attn_mask, self.compute_dtype)
        attn_mask = attn_mask[None, :, None, ...]

        return attn_mask

    def call(self, inputs, *args, **kwargs):
        attn_mask = self.attention_mask(inputs)

        outputs = inputs
        for b in self.blocks:
            outputs = b([outputs, attn_mask])

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
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
            'path_drop': self.path_drop
        })

        return config
