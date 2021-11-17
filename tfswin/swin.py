import numpy as np
import tensorflow as tf
from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from tfswin.drop import DropPath
from tfswin.mlp import MLP
from tfswin.norm import LayerNorm
from tfswin.winatt import WindowAttention


@register_keras_serializable(package='TFSwin')
class SwinBlock(layers.Layer):
    def __init__(self, num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., path_drop=0., **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=3)

        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop = drop
        self.attn_drop = attn_drop
        self.path_drop = path_drop

    @shape_type_conversion
    def build(self, input_shape):
        # noinspection PyAttributeOutsideInit
        self.length, self.channels = input_shape[1:]
        if None in {self.length, self.channels}:
            raise ValueError('Length and channel dimensions of the inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=3, axes={1: self.length, 2: self.channels})

        # noinspection PyAttributeOutsideInit
        self.size = int(self.length ** 0.5)
        if self.size ** 2 != self.length:
            raise ValueError('Height and width of the inputs should be equal.')

        if self.size <= self.window_size:
            self.shift_size = 0
            self.window_size = self.size
        if not 0 <= self.shift_size < self.window_size:
            raise ValueError('Shift size must be in range [0; window_size).')

        # noinspection PyAttributeOutsideInit
        self.norm1 = LayerNorm(name='norm1')

        # noinspection PyAttributeOutsideInit
        self.attn = WindowAttention(window_size=self.window_size, num_heads=self.num_heads, qkv_bias=self.qkv_bias,
                                    qk_scale=self.qk_scale, attn_mask=bool(self.shift_size), attn_drop=self.attn_drop,
                                    proj_drop=self.drop, name='attn')

        # noinspection PyAttributeOutsideInit
        self.drop_path = DropPath(self.path_drop)

        # noinspection PyAttributeOutsideInit
        self.norm2 = LayerNorm(name='norm2')

        # noinspection PyAttributeOutsideInit
        self.mlp = MLP(ratio=self.mlp_ratio, dropout=self.drop, name='mlp')

        super().build(input_shape)

    def attn_mask(self):
        img_mask = np.zeros([1, self.size, self.size, 1], 'float32')
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size, self.compute_dtype)[..., 0]
        attn_mask = mask_windows[:, None] - mask_windows[:, :, None]
        attn_mask = tf.cast(attn_mask == 0., self.compute_dtype) - 101.

        return attn_mask

    def call(self, inputs, *args, **kwargs):
        outputs = self.norm1(inputs)
        outputs = tf.reshape(outputs, [-1, self.size, self.size, self.channels])

        # Cyclic shift
        if self.shift_size > 0:
            outputs = tf.roll(outputs, [-self.shift_size, -self.shift_size], [1, 2])

        # Partition windows
        outputs = window_partition(outputs, self.window_size)

        # W-MSA/SW-MSA
        if self.shift_size:
            outputs = self.attn([outputs, self.attn_mask()])
        else:
            outputs = self.attn(outputs)

        # Merge windows
        outputs = window_reverse(outputs, self.size)

        # Reverse cyclic shift
        if self.shift_size > 0:
            outputs = tf.roll(outputs, [self.shift_size, self.shift_size], [1, 2])

        # FFN
        outputs = tf.reshape(outputs, [-1, self.length, self.channels])
        outputs = inputs + self.drop_path(outputs)
        outputs += self.drop_path(self.mlp(self.norm2(outputs)))

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

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


def window_partition(inputs, window_size, dtype=None, name=None):
    with tf.name_scope(name or 'window_partition'):
        inputs = tf.convert_to_tensor(inputs, dtype)

        if 4 != inputs.shape.rank:
            raise ValueError('Expecting inputs rank to be 4.')

        if None in set(inputs.shape[1:]):
            raise ValueError('Height, width and channel dimensions of the inputs should be defined. Found `None`.')

        height, width, channels = inputs.shape[1:]

        if height != width:
            raise ValueError('Height and width of the inputs should be equal.')
        num_windows = height // window_size

        outputs = tf.reshape(inputs, [-1, num_windows, window_size, num_windows, window_size, channels])
        outputs = tf.transpose(outputs, [0, 1, 3, 2, 4, 5])
        outputs = tf.reshape(outputs, [-1, window_size ** 2, channels])

        return outputs


def window_reverse(inputs, size, dtype=None, name=None):
    with tf.name_scope(name or 'window_reverse'):
        inputs = tf.convert_to_tensor(inputs, dtype)

        if 3 != inputs.shape.rank:
            raise ValueError('Expecting inputs rank to be 3.')

        if None in set(inputs.shape[1:]):
            raise ValueError('Length and channel dimensions of the inputs should be defined. Found `None`.')

        length, channels = inputs.shape[1:]

        window_size = int(length ** 0.5)
        if window_size ** 2 != length:
            raise ValueError('Length should be equal to window size ^ 2.')

        num_windows = size // window_size

        outputs = tf.reshape(inputs, [-1, num_windows, num_windows, window_size, window_size, channels])
        outputs = tf.transpose(outputs, [0, 1, 3, 2, 4, 5])
        outputs = tf.reshape(outputs, [-1, size, size, channels])

        return outputs
