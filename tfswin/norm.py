from keras import layers
from keras.saving.object_registration import register_keras_serializable


@register_keras_serializable(package='TFSwin')
class LayerNorm(layers.LayerNormalization):
    # Overload to use fused implementation

    def __init__(self, epsilon=1.001e-5, **kwargs):
        super().__init__(epsilon=epsilon, **kwargs)

    def _fused_can_be_used(self, ndims):
        if 'float32' != self.dtype:
            raise ValueError(
                f'Fused layer normalization is only supported when the variables dtype is '
                f'float32. Got dtype: {self.dtype}.')

        if self._compute_dtype not in ('float16', 'float16', 'float32', None):
            raise ValueError(
                f'Fused layer normalization is only supported when the compute dtype is '
                f'float16, bfloat16, or float32. Got dtype: {self._compute_dtype}.')

        if self.epsilon < 1.001e-5:
            raise ValueError(
                f'Fused layer normalization is not supported for epsilon {self.epsilon} (<1.001e-5).')

        axis = sorted(self.axis)
        if axis[-1] != ndims - 1 or axis[-1] - axis[0] != len(axis) - 1:
            raise ValueError(
                f'Fused layer normalization is not supported for axis {self.axis} and inputs rank {ndims}.')

        return True
