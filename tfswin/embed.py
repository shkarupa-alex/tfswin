from keras.src import layers
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable


@register_keras_serializable(package='TFSwin')
class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size, embed_dim, normalize, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=4)

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.normalize = normalize

    def build(self, input_shape):
        # noinspection PyAttributeOutsideInit
        self.proj = layers.Conv2D(
            self.embed_dim, kernel_size=self.patch_size, strides=self.patch_size, padding='same', name='proj', dtype=self.dtype_policy)
        self.proj.build(input_shape)

        if self.normalize:
            # noinspection PyAttributeOutsideInit
            self.norm = layers.LayerNormalization(epsilon=1.001e-5, name='norm', dtype=self.dtype_policy)
            self.norm.build((None, None, None, self.embed_dim))

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        outputs = self.proj(inputs)

        if self.normalize:
            outputs = self.norm(outputs)

        return outputs

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
