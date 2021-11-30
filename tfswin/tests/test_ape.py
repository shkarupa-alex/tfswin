import tensorflow as tf
from keras import keras_parameterized, testing_utils
from tfswin.ape import AbsoluteEmbedding


@keras_parameterized.run_all_keras_modes
class TestAbsoluteEmbedding(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            AbsoluteEmbedding,
            kwargs={'pretrain_size': 56},
            input_shape=[2, 56, 56, 3],
            input_dtype='float32',
            expected_output_shape=[None, 56, 56, 3],
            expected_output_dtype='float32'
        )
        testing_utils.layer_test(
            AbsoluteEmbedding,
            kwargs={'pretrain_size': 56},
            input_shape=[2, 112, 112, 3],
            input_dtype='float32',
            expected_output_shape=[None, 112, 112, 3],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
