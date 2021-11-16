import numpy as np
import os
import tensorflow as tf
from keras import keras_parameterized, testing_utils
from ..embed import PatchEmbedding


@keras_parameterized.run_all_keras_modes
class TestPatchEmbedding(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            PatchEmbedding,
            kwargs={'patch_size': 4, 'embed_dim': 2, 'normalize': False},
            input_shape=[2, 12, 12, 3],
            input_dtype='float32',
            expected_output_shape=[None, 9, 2],
            expected_output_dtype='float32'
        )
        testing_utils.layer_test(
            PatchEmbedding,
            kwargs={'patch_size': 3, 'embed_dim': 2, 'normalize': True},
            input_shape=[2, 12, 12, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 2],
            expected_output_dtype='float32'
        )

    def test_value(self):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        inputs = np.load(f'{data_dir}/_patch_embed_input.npy').transpose([0, 2, 3, 1])
        targets = np.load(f'{data_dir}/_patch_embed_output.npy')
        layer = PatchEmbedding(4, 96, True)
        layer(inputs)  # build
        layer.set_weights([
            np.load(f'{data_dir}/_patch_embed_proj_weight.npy').transpose([2, 3, 1, 0]),
            np.load(f'{data_dir}/_patch_embed_proj_bias.npy').T,
            np.load(f'{data_dir}/_patch_embed_norm_weight.npy').T,
            np.load(f'{data_dir}/_patch_embed_norm_bias.npy').T,
        ])
        outputs = self.evaluate(layer(inputs))
        self.assertLess(np.abs(targets - outputs).max(), 2.39e-6)


if __name__ == '__main__':
    tf.test.main()
