import numpy as np
import tensorflow as tf
from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from tfswin.norm import LayerNorm


@register_keras_serializable(package='TFSwin')
class PatchMerging(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=3)

        self.length = None
        self.channels = None
        self.size = None

    @shape_type_conversion
    def build(self, input_shape):
        self.length, self.channels = input_shape[1:]
        if None in {self.length, self.channels}:
            raise ValueError('Length and channel dimensions of the inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=3, axes={1: self.length, 2: self.channels})

        self.size = int(self.length ** 0.5)
        if self.size ** 2 != self.length:
            raise ValueError('Height and width of the inputs should be equal.')
        if self.size % 2:
            raise ValueError('Height and width should be evenly dividable by 2.')

        # noinspection PyAttributeOutsideInit
        self.norm = LayerNorm(name='norm')

        # noinspection PyAttributeOutsideInit
        self.reduction = layers.Dense(self.channels * 2, use_bias=False, name='reduction')

        indices = np.arange(0, self.length).reshape((self.size, self.size))
        # noinspection PyAttributeOutsideInit
        self.indices = np.stack([
            indices[0::2, 0::2],  # B H/2 W/2 C
            indices[1::2, 0::2],  # B H/2 W/2 C
            indices[0::2, 1::2],  # B H/2 W/2 C
            indices[1::2, 1::2]  # B H/2 W/2 C
        ], axis=-1).ravel()

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        outputs = tf.gather(inputs, self.indices, axis=1)
        outputs = tf.reshape(outputs, [-1, self.length // 4, self.channels * 4])

        outputs = self.norm(outputs)
        outputs = self.reduction(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        out_length = None if self.length is None else self.length // 4
        out_channels = None if self.channels is None else self.channels * 2

        return [input_shape[0], out_length, out_channels]
