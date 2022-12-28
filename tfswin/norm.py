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

    # def _fused_can_be_used(self, ndims):
    #     """Returns false if fused implementation cannot be used.
    #
    #     Check if the axis is contiguous and can be collapsed into the last axis.
    #     The self.axis is assumed to have no duplicates.
    #     """
    #     axis = sorted(self.axis)
    #     can_use_fused = False
    #
    #     if axis[-1] == ndims - 1 and axis[-1] - axis[0] == len(axis) - 1:
    #         can_use_fused = True
    #
    #     # fused_batch_norm will silently raise epsilon to be at least 1.001e-5,
    #     # so we cannot used the fused version if epsilon is below that value.
    #     # Also, the variable dtype must be float32, as fused_batch_norm only
    #     # supports float32 variables.
    #     if self.epsilon < 1.001e-5 or self.dtype != "float32":
    #         can_use_fused = False
    #
    #     return can_use_fused
