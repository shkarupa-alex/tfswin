import tensorflow as tf
from keras.testing_infra import test_combinations, test_utils
from tfswin.ape import AbsoluteEmbedding


@test_combinations.run_all_keras_modes
class TestAbsoluteEmbedding(test_combinations.TestCase):
    def test_layer(self):
        test_utils.layer_test(
            AbsoluteEmbedding,
            kwargs={'pretrain_size': 56},
            input_shape=[2, 56, 56, 3],
            input_dtype='float32',
            expected_output_shape=[None, 56, 56, 3],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            AbsoluteEmbedding,
            kwargs={'pretrain_size': 56},
            input_shape=[2, 112, 112, 3],
            input_dtype='float32',
            expected_output_shape=[None, 112, 112, 3],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
