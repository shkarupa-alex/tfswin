import tensorflow as tf
from keras.testing_infra import test_combinations, test_utils
from tfswin.merge import PatchMerging


@test_combinations.run_all_keras_modes
class TestPatchMerging(test_combinations.TestCase):
    def test_layer(self):
        test_utils.layer_test(
            PatchMerging,
            kwargs={'swin_v2': False},
            input_shape=[2, 12, 12, 4],
            input_dtype='float32',
            expected_output_shape=[None, 6, 6, 8],
            expected_output_dtype='float32'
        )

        test_utils.layer_test(
            PatchMerging,
            kwargs={'swin_v2': True},
            input_shape=[2, 11, 13, 3],
            input_dtype='float32',
            expected_output_shape=[None, 6, 7, 6],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
