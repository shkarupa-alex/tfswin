import numpy as np
import tensorflow as tf
from keras.src import initializers, layers
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable
from tfswin.window import window_partition_fused, window_reverse_fused


@register_keras_serializable(package='TFSwin')
class WindowAttention(layers.Layer):
    def __init__(self, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 window_pretrain=0, swin_v2=False, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            InputSpec(ndim=4), InputSpec(ndim=0, dtype='int32'), InputSpec(ndim=1, dtype='int32'),
            InputSpec(ndim=5)]

        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.window_pretrain = window_pretrain
        self.swin_v2 = swin_v2

    def build(self, input_shape):
        # noinspection PyAttributeOutsideInit
        self.channels = input_shape[0][-1]
        if self.channels is None:
            raise ValueError('Channel dimensions of the inputs should be defined. Found `None`.')

        qkv_bias = not self.swin_v2 and self.qkv_bias
        # noinspection PyAttributeOutsideInit
        self.qkv = layers.Dense(self.channels * 3, use_bias=qkv_bias, name='qkv', dtype=self.dtype_policy)
        self.qkv.build(input_shape[0])

        if self.swin_v2:
            # noinspection PyAttributeOutsideInit
            self.scale = self.add_weight(
                name='logit_scale',
                shape=[self.num_heads, 1, 1],
                initializer=initializers.Constant(np.log(10.)),
                trainable=True,
                dtype=self.dtype)

            # noinspection PyAttributeOutsideInit
            self.cpb0 = layers.Dense(512, activation='relu', name='cpb_mlp.0', dtype=self.dtype_policy)
            self.cpb0.build([1, None, None, 2])
            self.cpb1 = layers.Dense(self.num_heads, activation='sigmoid', use_bias=False, name=f'cpb_mlp.2', dtype=self.dtype_policy)
            self.cpb1.build([1, None, None, 512])

            # noinspection PyAttributeOutsideInit
            self.q_bias = None
            # noinspection PyAttributeOutsideInit
            self.v_bias = None
            if self.qkv_bias:
                self.q_bias = self.add_weight(
                    name='q_bias',
                    shape=[self.channels],
                    initializer='zeros',
                    trainable=True,
                    dtype=self.dtype)
                self.v_bias = self.add_weight(
                    name='v_bias',
                    shape=[self.channels],
                    initializer='zeros',
                    trainable=True,
                    dtype=self.dtype)
        else:
            # noinspection PyAttributeOutsideInit
            self.scale = self.qk_scale or (self.channels // self.num_heads) ** -0.5

            # noinspection PyAttributeOutsideInit
            self.relative_bias = self.add_weight(
                name='relative_position_bias_table',
                shape=[(2 * self.window_pretrain - 1) ** 2, self.num_heads],
                initializer=initializers.TruncatedNormal(stddev=0.02),
                trainable=True,
                dtype=self.dtype)

        # noinspection PyAttributeOutsideInit
        self.drop_attn = layers.Dropout(self.attn_drop, dtype=self.dtype_policy)

        # noinspection PyAttributeOutsideInit
        self.proj = layers.Dense(self.channels, name='proj', dtype=self.dtype_policy)
        self.proj.build(input_shape[0])

        # noinspection PyAttributeOutsideInit
        self.drop_proj = layers.Dropout(self.proj_drop, dtype=self.dtype_policy)

        super().build(input_shape)

    def relative_table(self, window_size):
        offset = tf.range(1 - window_size, window_size)
        offset = tf.cast(offset, self.compute_dtype)
        offset = tf.stack(tf.meshgrid(offset, offset, indexing='ij'))
        offset = tf.transpose(offset, [1, 2, 0])[None]

        window = self.window_pretrain if self.window_pretrain > 0 else window_size

        offset *= 8. / (tf.cast(window, self.compute_dtype) - 1.)
        offset = tf.sign(offset) * tf.math.log1p(tf.abs(offset)) / np.log(8)

        return offset

    def with_mask(self, attn, mask, length):
        mask_windows = tf.shape(mask)[1]
        attn = tf.reshape(attn, shape=[-1, mask_windows, self.num_heads, length, length])
        attn += mask
        attn = tf.reshape(attn, shape=[-1, self.num_heads, length, length])

        return attn

    def call(self, inputs, **kwargs):
        inputs, window_size, relative_index, attention_mask = inputs
        height, width = tf.unstack(tf.shape(inputs)[1:3])
        length = window_size ** 2

        qkv = self.qkv(inputs)
        if self.swin_v2 and self.qkv_bias:
            k_bias = tf.zeros_like(self.v_bias, self.compute_dtype)
            qkv_bias = tf.concat([self.q_bias, k_bias, self.v_bias], axis=0)
            qkv = tf.nn.bias_add(qkv, qkv_bias)

        # QKV heads partition - fused with windows partitioning
        # qkv = tf.reshape(qkv, [-1, length, 3, self.num_heads, self.channels // self.num_heads])
        # qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])

        qkv = window_partition_fused(qkv, height, width, window_size, self.num_heads)

        q, k, v = tf.unstack(qkv, 3)
        if self.swin_v2:
            scale = tf.minimum(self.scale, np.log(1. / .01))
            scale = tf.exp(scale)
            q = tf.math.l2_normalize(q, axis=-1, epsilon=1.55e-5)
            k = tf.math.l2_normalize(k, axis=-1, epsilon=1.55e-5)
        else:
            scale = self.scale
        q *= scale
        attn = tf.matmul(q, k, transpose_b=True)

        if self.swin_v2:
            relative_bias = self.cpb0(self.relative_table(window_size))
            relative_bias = self.cpb1(relative_bias)
            relative_bias = tf.reshape(relative_bias, [-1, self.num_heads])
            bias = tf.gather(relative_bias, relative_index) * 16.
        else:
            bias = tf.gather(self.relative_bias, relative_index)
        bias = tf.reshape(bias, [length, length, -1])
        bias = tf.transpose(bias, perm=[2, 0, 1])
        attn = attn + bias[None]

        attn = self.with_mask(attn, attention_mask, length)

        attn = tf.nn.softmax(attn)
        attn = self.drop_attn(attn)

        outputs = tf.matmul(attn, v)

        # V heads merge - fused with windows merging
        # outputs = tf.transpose(outputs, perm=[0, 2, 1, 3])
        # outputs = tf.reshape(outputs, [-1, length, self.channels])

        outputs = window_reverse_fused(outputs, height, width, window_size, self.num_heads)

        outputs = self.proj(outputs)
        outputs = self.drop_proj(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()

        config.update({
            'num_heads': self.num_heads,
            'qkv_bias': self.qkv_bias,
            'qk_scale': self.qk_scale,
            'attn_drop': self.attn_drop,
            'proj_drop': self.proj_drop,
            'window_pretrain': self.window_pretrain,
            'swin_v2': self.swin_v2
        })

        return config
