import tensorflow as tf
import warnings
from keras import layers
from keras.utils.generic_utils import register_keras_serializable


@register_keras_serializable(package='TFSwin')
class LayerNorm(layers.LayerNormalization):
    # Overload defaults and casting to use fused implementation

    def __init__(self, epsilon=1.001e-5, dtype='float32', **kwargs):
        kwargs['autocast'] = False
        super().__init__(epsilon=epsilon, dtype=dtype, **kwargs)

    def call(self, inputs, *args, **kwargs):
        outputs = tf.cast(inputs, 'float32')

        outputs = super().call(outputs)
        if not self._fused:
            warnings.warn(f'Layer {self.name} will use an inefficient implementation.')

        if inputs.dtype == tf.dtypes.float16:
            outputs = tf.clip_by_value(outputs, tf.dtypes.float16.min, tf.dtypes.float16.max)
        outputs = tf.cast(outputs, inputs.dtype)

        return outputs

    def compute_output_signature(self, input_signature):
        return input_signature
