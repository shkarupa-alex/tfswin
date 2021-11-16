import tensorflow as tf
from keras import keras_parameterized, testing_utils
from ..ape import AbsolutePositionEmbedding


@keras_parameterized.run_all_keras_modes
class TestAbsolutePositionEmbedding(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            AbsolutePositionEmbedding,
            kwargs={},
            input_shape=[2, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 3],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
