from keras import initializers, layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='TFSwin')
class AbsoluteEmbedding(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=3)

    @shape_type_conversion
    def build(self, input_shape):
        num_patches, embed_dim = input_shape[-2:]
        if None in {num_patches, embed_dim}:
            raise ValueError('Number of patches and embedding dimension of the inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=3, axes={1: num_patches, 2: embed_dim})

        # noinspection PyAttributeOutsideInit
        self.embedding = self.add_weight(
            'embedding',
            shape=[1, num_patches, embed_dim],
            initializer=initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            dtype=self.dtype)

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        return inputs + self.embedding

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape
