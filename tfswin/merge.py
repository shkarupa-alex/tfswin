import math
import numpy as np
import tensorflow as tf
from keras.src import layers
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable


@register_keras_serializable(package='TFSwin')
class PatchMerging(layers.Layer):
    def __init__(self, swin_v2=False, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=4)

        self.swin_v2 = swin_v2

    def build(self, input_shape):
        # noinspection PyAttributeOutsideInit
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError('Channel dimensions of the inputs should be defined. Found `None`.')
        self.input_spec = InputSpec(ndim=4, axes={-1: self.channels})

        # Channel permutation after space-to-depth
        # Required due to unusual concatenation order in orignal version
        perm = np.arange(self.channels * 4).reshape((4, -1))
        perm[[1, 2]] = perm[[2, 1]]

        # noinspection PyAttributeOutsideInit
        self.perm = perm.ravel()

        # noinspection PyAttributeOutsideInit
        self.norm = layers.LayerNormalization(epsilon=1.001e-5, name='norm', dtype=self.dtype_policy)

        # noinspection PyAttributeOutsideInit
        self.reduction = layers.Dense(self.channels * 2, use_bias=False, name='reduction', dtype=self.dtype_policy)

        if self.swin_v2:
            self.reduction.build((None, None, None, self.channels * 4))
            self.norm.build((None, None, None, self.channels * 2))
        else:
            self.norm.build((None, None, None, self.channels * 4))
            self.reduction.build((None, None, None, self.channels * 4))

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        def _pad(value):
            return None if value is None else value + value % 2

        height_width = tf.shape(inputs)[1:3]
        hpad, wpad = tf.unstack(height_width % 2)
        paddings = [[0, 0], [0, hpad], [0, wpad], [0, 0]]
        outputs = tf.pad(inputs, paddings)
        outputs.set_shape((inputs.shape[0], _pad(inputs.shape[1]), _pad(inputs.shape[2]), inputs.shape[3]))

        outputs = tf.nn.space_to_depth(outputs, 2)
        outputs = tf.gather(outputs, self.perm, batch_dims=-1)

        if self.swin_v2:
            outputs = self.reduction(outputs)
            outputs = self.norm(outputs)
        else:
            outputs = self.norm(outputs)
            outputs = self.reduction(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        def _scale(value):
            return None if value is None else math.ceil(value / 2)

        return input_shape[0], _scale(input_shape[1]), _scale(input_shape[2]), self.channels * 2

    def get_config(self):
        config = super().get_config()
        config.update({'swin_v2': self.swin_v2})

        return config
