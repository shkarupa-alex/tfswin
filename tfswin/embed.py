from keras import layers
from keras.saving.object_registration import register_keras_serializable
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

    @shape_type_conversion
    def build(self, input_shape):
        # noinspection PyAttributeOutsideInit
        self.proj = layers.Conv2D(
            self.embed_dim, kernel_size=self.patch_size, strides=self.patch_size, padding='same', name='proj')

        if self.normalize:
            # noinspection PyAttributeOutsideInit
            self.norm = LayerNorm(name='norm')

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        outputs = self.proj(inputs)

        if self.normalize:
            outputs = self.norm(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return self.proj.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim,
            'normalize': self.normalize,
        })

        return config
