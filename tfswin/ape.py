import tensorflow as tf
from keras import initializers, layers
from keras.saving.object_registration import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='TFSwin')
class AbsoluteEmbedding(layers.Layer):
    def __init__(self, pretrain_size, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.pretrain_size = pretrain_size

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=4, axes={-1: channels})

        # noinspection PyAttributeOutsideInit
        self.embedding = self.add_weight(
            'embedding',
            shape=[1, self.pretrain_size, self.pretrain_size, channels],
            initializer=initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            dtype=self.dtype)

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        new_size = tf.shape(inputs)[1:3]
        embeddings = tf.image.resize(self.embedding, new_size, method=tf.image.ResizeMethod.BICUBIC)
        embeddings = tf.cast(embeddings, inputs.dtype)

        return inputs + embeddings

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({'pretrain_size': self.pretrain_size})

        return config
