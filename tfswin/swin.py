import tensorflow as tf
from keras import layers
from keras.saving.object_registration import register_keras_serializable
from keras.utils.control_flow_util import smart_cond
from keras.utils.tf_utils import shape_type_conversion
from tfswin.drop import DropPath
from tfswin.mlp import MLP
from tfswin.norm import LayerNorm
from tfswin.winatt import WindowAttention


@register_keras_serializable(package='TFSwin')
class SwinBlock(layers.Layer):
    def __init__(self, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., path_drop=0.,
                 window_pretrain=0, swin_v2=False, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4), layers.InputSpec(ndim=0, dtype='int32'), layers.InputSpec(ndim=0, dtype='int32'),
            layers.InputSpec(ndim=1, dtype='int32'), layers.InputSpec(ndim=5)]
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop = drop
        self.attn_drop = attn_drop
        self.path_drop = path_drop
        self.window_pretrain = window_pretrain
        self.swin_v2 = swin_v2

    @shape_type_conversion
    def build(self, input_shape):
        norm_init = 'zeros' if self.swin_v2 else 'ones'

        # noinspection PyAttributeOutsideInit
        self.norm1 = LayerNorm(gamma_initializer=norm_init, name='norm1')

        # noinspection PyAttributeOutsideInit
        self.attn = WindowAttention(num_heads=self.num_heads, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                                    attn_drop=self.attn_drop, proj_drop=self.drop, window_pretrain=self.window_pretrain,
                                    swin_v2=self.swin_v2, name='attn')

        # noinspection PyAttributeOutsideInit
        self.drop_path = DropPath(self.path_drop)

        # noinspection PyAttributeOutsideInit
        self.norm2 = LayerNorm(gamma_initializer=norm_init, name='norm2')

        # noinspection PyAttributeOutsideInit
        self.mlp = MLP(ratio=self.mlp_ratio, dropout=self.drop, name='mlp')

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        inputs, shift_size, window_size, relative_index, attention_mask = inputs
        height, width = tf.unstack(tf.shape(inputs)[1:3])

        if self.swin_v2:
            outputs = inputs
        else:
            outputs = self.norm1(inputs)

        h_pad = (window_size - height % window_size) % window_size
        w_pad = (window_size - width % window_size) % window_size
        paddings = [[0, 0], [0, h_pad], [0, w_pad], [0, 0]]
        outputs = tf.pad(outputs, paddings)

        # Cyclic shift
        with_shift = shift_size > 0
        outputs = smart_cond(
            with_shift,
            lambda: tf.roll(outputs, [-shift_size, -shift_size], [1, 2]),
            lambda: tf.identity(outputs)
        )

        # W-MSA/SW-MSA
        outputs = self.attn([outputs, window_size, relative_index, attention_mask])

        # Reverse cyclic shift
        outputs = smart_cond(
            with_shift,
            lambda: tf.roll(outputs, [shift_size, shift_size], [1, 2]),
            lambda: tf.identity(outputs)
        )

        outputs = outputs[:, :height, :width]

        # FFN
        if self.swin_v2:
            outputs = inputs + self.drop_path(self.norm1(outputs))
            outputs += self.drop_path(self.norm2(self.mlp(outputs)))
        else:
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
            'mlp_ratio': self.mlp_ratio,
            'qkv_bias': self.qkv_bias,
            'qk_scale': self.qk_scale,
            'drop': self.drop,
            'attn_drop': self.attn_drop,
            'path_drop': self.path_drop,
            'window_pretrain': self.window_pretrain,
            'swin_v2': self.swin_v2
        })

        return config
