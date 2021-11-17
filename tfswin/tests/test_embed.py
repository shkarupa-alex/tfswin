import tensorflow as tf
from keras import keras_parameterized, testing_utils
from tfswin.embed import PatchEmbedding


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


if __name__ == '__main__':
    tf.test.main()
