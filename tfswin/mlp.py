from keras import activations, layers
from keras.saving.object_registration import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='TFSwin')
class MLP(layers.Layer):
    def __init__(self, ratio, dropout, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.ratio = ratio
        self.dropout = dropout

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=4, axes={-1: channels})

        # noinspection PyAttributeOutsideInit
        self.fc1 = layers.Dense(int(channels * self.ratio), name='fc1')

        # noinspection PyAttributeOutsideInit
        self.fc2 = layers.Dense(channels, name='fc2')

        # noinspection PyAttributeOutsideInit
        self.drop = layers.Dropout(self.dropout)

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        outputs = self.fc1(inputs)
        outputs = activations.gelu(outputs)
        outputs = self.drop(outputs)
        outputs = self.fc2(outputs)
        outputs = self.drop(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'ratio': self.ratio,
            'dropout': self.dropout
        })

        return config
