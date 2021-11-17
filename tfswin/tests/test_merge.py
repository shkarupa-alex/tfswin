import tensorflow as tf
from keras import keras_parameterized, testing_utils
from tfswin.merge import PatchMerging


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


if __name__ == '__main__':
    tf.test.main()
