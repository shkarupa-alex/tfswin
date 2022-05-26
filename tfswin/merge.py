import math
import tensorflow as tf
from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from tfswin.norm import LayerNorm


@register_keras_serializable(package='TFSwin')
class PatchMerging(layers.Layer):
    def __init__(self, swin_v2=False, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.swin_v2 = swin_v2

    @shape_type_conversion
    def build(self, input_shape):
        # noinspection PyAttributeOutsideInit
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError('Channel dimensions of the inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=4, axes={-1: self.channels})

        # noinspection PyAttributeOutsideInit
        self.norm = LayerNorm(name='norm')

        # noinspection PyAttributeOutsideInit
        self.reduction = layers.Dense(self.channels * 2, use_bias=False, name='reduction')

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        paddings = [[0, 0], [0, 1], [0, 1], [0, 0]]
        outputs = tf.pad(inputs, paddings)

        slice00 = outputs[:, 0:-1:2, 0:-1:2, :]  # B H/2 W/2 C
        slice10 = outputs[:, 1::2, 0:-1:2, :]  # B H/2 W/2 C
        slice01 = outputs[:, 0:-1:2, 1::2, :]  # B H/2 W/2 C
        slice11 = outputs[:, 1::2, 1::2, :]  # B H/2 W/2 C
        outputs = tf.concat([slice00, slice10, slice01, slice11], axis=-1)

        if self.swin_v2:
            outputs = self.reduction(outputs)
            outputs = self.norm(outputs)
        else:
            outputs = self.norm(outputs)
            outputs = self.reduction(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        def _scale(value):
            return None if value is None else math.ceil(value / 2)

        return input_shape[0], _scale(input_shape[1]), _scale(input_shape[2]), self.channels * 2

    def get_config(self):
        config = super().get_config()
        config.update({'swin_v2': self.swin_v2})

        return config
