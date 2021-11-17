import tensorflow as tf
from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from tfswin.norm import LayerNorm


@register_keras_serializable(package='TFSwin')
class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size, embed_dim, normalize, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.normalize = normalize

        self.num_patches = None

    @shape_type_conversion
    def build(self, input_shape):
        height, width, channels = input_shape[1:]
        if None in {height, width, channels}:
            raise ValueError('Height, width and channel dimensions of the inputs should be defined. Found `None`.')
        if height % self.patch_size or width % self.patch_size:
            raise ValueError('Height and width should be evenly dividable by corresponding patch size.')
        self.input_spec = layers.InputSpec(ndim=4, axes={1: height, 2: width, 3: channels})

        self.num_patches = (height // self.patch_size) * (width // self.patch_size)

        # noinspection PyAttributeOutsideInit
        self.proj = layers.Conv2D(self.embed_dim, kernel_size=self.patch_size, strides=self.patch_size, name='proj')

        self.norm = None
        if self.normalize:
            # noinspection PyAttributeOutsideInit
            self.norm = LayerNorm(name='norm')

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        outputs = self.proj(inputs)
        outputs = tf.reshape(outputs, [-1, self.num_patches, self.embed_dim])

        if self.norm is not None:
            outputs = self.norm(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.num_patches, self.embed_dim]

    def get_config(self):
        config = super().get_config()
        config.update({
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim,
            'normalize': self.normalize,
        })

        return config
