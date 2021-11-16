import numpy as np
import tensorflow as tf
from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='TFSwin')
class WindowAttention(layers.Layer):
    def __init__(self, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=3)

        self.window_size = window_size
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop

    @shape_type_conversion
    def build(self, input_shape):
        # noinspection PyAttributeOutsideInit
        self.length, self.channels = input_shape[1:]
        if None in {self.length, self.channels}:
            raise ValueError('Length and channel dimensions of the inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=3, axes={1: self.length, 2: self.channels})

        # noinspection PyAttributeOutsideInit
        self.scale = self.qk_scale or (self.channels // self.num_heads) ** -0.5

        # noinspection PyAttributeOutsideInit
        self.qkv = layers.Dense(self.channels * 3, use_bias=self.qkv_bias, name='qkv')

        self.drop_att = layers.Dropout(self.attn_drop)
        self.proj = layers.Dense(self.channels, name='proj')
        self.drop_proj = layers.Dropout(self.proj_drop)

        coords_h = np.arange(self.window_size)
        coords_w = np.arange(self.window_size)
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1).astype(np.int64)
        self.relative_position_index = relative_position_index.ravel()
        self.relative_position_bias_table = self.add_weight(
            'relative_position_bias_table',
            shape=((2 * self.window_size - 1) ** 2, self.num_heads),
            initializer='zero', trainable=True)

        self.built = True

    def call(self, inputs, mask=None, **kwargs):
        B_, N, C = inputs.get_shape().as_list()
        qkv = self.qkv(inputs)  # B, N, 3 * C == B, N, 3, self.num_heads, C // self.num_heads
        qkv = tf.reshape(qkv, [-1, self.length, 3, self.num_heads, self.channels // self.num_heads])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = tf.unstack(qkv, 3, axis=0)

        q = q * self.scale
        attn = tf.matmul(q, k, transpose_b=True)
        relative_position_bias = tf.gather(self.relative_position_bias_table, self.relative_position_index)
        relative_position_bias = tf.reshape(relative_position_bias, shape=[
            self.window_size ** 2, self.window_size ** 2, -1])
        relative_position_bias = tf.transpose(
            relative_position_bias, perm=[2, 0, 1])
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.get_shape()[0]  # tf.shape(mask)[0]
            attn = tf.reshape(attn, shape=[-1, nW, self.num_heads, N, N]) + tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32)
            attn = tf.reshape(attn, shape=[-1, self.num_heads, N, N])
            attn = tf.nn.softmax(attn, axis=-1)
        else:
            attn = tf.nn.softmax(attn, axis=-1)

        attn = self.drop_att(attn)

        outputs = tf.transpose(tf.matmul(attn, v), perm=[0, 2, 1, 3])
        outputs = tf.reshape(outputs, shape=[-1, N, C])

        outputs = self.proj(outputs)
        outputs = self.drop_proj(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()

        config.update({
            'window_size': self.window_size,
            'num_heads': self.num_heads,
            'qkv_bias': self.qkv_bias,
            'qk_scale': self.qk_scale,
            'attn_drop': self.attn_drop,
            'proj_drop': self.proj_drop,
        })

        return config
