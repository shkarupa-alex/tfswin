import tensorflow as tf
from keras.testing_infra import test_combinations, test_utils
from tfswin.mlp import MLP


@test_combinations.run_all_keras_modes
class TestMLP(test_combinations.TestCase):
    def test_layer(self):
        test_utils.layer_test(
            MLP,
            kwargs={'ratio': 0.5, 'dropout': 0.},
            input_shape=[2, 4, 4, 3],
            input_dtype='float32',
            expected_output_shape=[None, 4, 4, 3],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            MLP,
            kwargs={'ratio': 1.5, 'dropout': 0.2},
            input_shape=[2, 4, 4, 3],
            input_dtype='float32',
            expected_output_shape=[None, 4, 4, 3],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
