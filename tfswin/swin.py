import numpy as np
import tensorflow as tf
from keras import layers
from keras.utils.control_flow_util import smart_cond
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from tfswin.drop import DropPath
from tfswin.mlp import MLP
from tfswin.norm import LayerNorm
from tfswin.winatt import WindowAttention
from tfswin.window import window_partition, window_reverse


@register_keras_serializable(package='TFSwin')
class SwinBlock(layers.Layer):
    def __init__(self, num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop=0., attn_drop=0., path_drop=0., **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [layers.InputSpec(ndim=4), layers.InputSpec(ndim=5)]

        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop = drop
        self.attn_drop = attn_drop
        self.path_drop = path_drop

        if not 0 <= self.shift_size < self.window_size:
            raise ValueError('Shift size must be in range [0; window_size).')

    @shape_type_conversion
    def build(self, input_shape):
        # noinspection PyAttributeOutsideInit
        self.norm1 = LayerNorm(name='norm1')

        # noinspection PyAttributeOutsideInit
        self.attn = WindowAttention(window_size=self.window_size, num_heads=self.num_heads, qkv_bias=self.qkv_bias,
                                    qk_scale=self.qk_scale, attn_drop=self.attn_drop, proj_drop=self.drop, name='attn')

        # noinspection PyAttributeOutsideInit
        self.drop_path = DropPath(self.path_drop)

        # noinspection PyAttributeOutsideInit
        self.norm2 = LayerNorm(name='norm2')

        # noinspection PyAttributeOutsideInit
        self.mlp = MLP(ratio=self.mlp_ratio, dropout=self.drop, name='mlp')

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        inputs, mask = inputs
        height, width = tf.unstack(tf.shape(inputs)[1:3])

        min_size = tf.minimum(height, width)
        shift_size, window_size = smart_cond(
            tf.less_equal(min_size, self.window_size),
            lambda: (0, min_size),
            lambda: (self.shift_size, self.window_size))
        with_shift = tf.greater(shift_size, 0)

        outputs = self.norm1(inputs)

        h_pad = (self.window_size - height % self.window_size) % self.window_size
        w_pad = (self.window_size - width % self.window_size) % self.window_size
        paddings = [[0, 0], [0, h_pad], [0, w_pad], [0, 0]]
        outputs = tf.pad(outputs, paddings)
        padded_height, padded_width = height + h_pad, width + w_pad

        # Cyclic shift
        outputs = smart_cond(
            with_shift,
            lambda: tf.roll(outputs, [-shift_size, -shift_size], [1, 2]),
            lambda: tf.identity(outputs))

        # Partition windows
        outputs = window_partition(outputs, padded_height, padded_width, window_size, self.compute_dtype)

        # W-MSA/SW-MSA
        outputs = self.attn([outputs, mask, with_shift])

        # Merge windows
        outputs = window_reverse(outputs, padded_height, padded_width, window_size, self.compute_dtype)

        # Reverse cyclic shift
        outputs = smart_cond(
            with_shift,
            lambda: tf.roll(outputs, [shift_size, shift_size], [1, 2]),
            lambda: tf.identity(outputs)
        )

        outputs = outputs[:, :height, :width, ...]

        # FFN
        outputs = inputs + self.drop_path(outputs)
        outputs += self.drop_path(self.mlp(self.norm2(outputs)))

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'window_size': self.window_size,
            'shift_size': self.shift_size,
            'mlp_ratio': self.mlp_ratio,
            'qkv_bias': self.qkv_bias,
            'qk_scale': self.qk_scale,
            'drop': self.drop,
            'attn_drop': self.attn_drop,
            'path_drop': self.path_drop,
        })

        return config
