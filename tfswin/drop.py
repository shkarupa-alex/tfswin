import tensorflow as tf
from keras.src import backend, layers
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable


@register_keras_serializable(package='TFSwin')
class DropPath(layers.Layer):
    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=1)

        if not 0. <= rate <= 1.:
            raise ValueError(f'Invalid value {rate} received for `rate`. Expected a value between 0 and 1.')

        self.rate = rate

    def call(self, inputs, training=False, **kwargs):
        if 0. == self.rate or not training:
            return inputs

        return self.drop(inputs)

    def drop(self, inputs):
        keep = 1.0 - self.rate
        batch = tf.shape(inputs)[0]
        shape = [batch] + [1] * (inputs.shape.rank - 1)

        random = tf.random.uniform(shape, dtype=self.compute_dtype) <= keep
        random = tf.cast(random, self.compute_dtype) / keep

        outputs = inputs * random

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({'rate': self.rate})

        return config