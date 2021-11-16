import numpy as np
import os
import tensorflow as tf
from keras import keras_parameterized, testing_utils
from ..merge import PatchMerging


@keras_parameterized.run_all_keras_modes
class TestPatchMerging(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            PatchMerging,
            kwargs={},
            input_shape=[2, 12 * 12, 4],
            input_dtype='float32',
            expected_output_shape=[None, 6 * 6, 8],
            expected_output_dtype='float32'
        )

    def test_value(self):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        inputs = np.load(f'{data_dir}/_patch_merge_input.npy')
        targets = np.load(f'{data_dir}/_patch_merge_output.npy')
        layer = PatchMerging()
        layer(inputs)  # build
        layer.set_weights([
            np.load(f'{data_dir}/_patch_merge_norm_weight.npy').T,
            np.load(f'{data_dir}/_patch_merge_norm_bias.npy').T,
            np.load(f'{data_dir}/_patch_merge_dense_weight.npy').T,
        ])
        outputs = self.evaluate(layer(inputs))
        self.assertLess(np.abs(targets - outputs).max(), 1.53e-5)


if __name__ == '__main__':
    tf.test.main()
