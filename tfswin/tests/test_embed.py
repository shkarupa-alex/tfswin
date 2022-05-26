import tensorflow as tf
from keras.testing_infra import test_combinations, test_utils
from tfswin.embed import PatchEmbedding


@test_combinations.run_all_keras_modes
class TestPatchEmbedding(test_combinations.TestCase):
    def test_layer(self):
        test_utils.layer_test(
            PatchEmbedding,
            kwargs={'patch_size': 4, 'embed_dim': 2, 'normalize': False},
            input_shape=[2, 12, 12, 3],
            input_dtype='float32',
            expected_output_shape=[None, 3, 3, 2],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            PatchEmbedding,
            kwargs={'patch_size': 3, 'embed_dim': 2, 'normalize': True},
            input_shape=[2, 12, 12, 3],
            input_dtype='float32',
            expected_output_shape=[None, 4, 4, 2],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
