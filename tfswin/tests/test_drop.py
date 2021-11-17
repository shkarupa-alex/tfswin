import tensorflow as tf
from keras import keras_parameterized, testing_utils
from tfswin.drop import DropPath


@keras_parameterized.run_all_keras_modes
class TestDropPath(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            DropPath,
            kwargs={'rate': 0.},
            input_shape=[2, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 3],
            expected_output_dtype='float32'
        )
        testing_utils.layer_test(
            DropPath,
            kwargs={'rate': 0.2},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 3],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
