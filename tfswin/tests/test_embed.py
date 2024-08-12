from keras.src import testing
from tfswin.embed import PatchEmbedding


class TestPatchEmbedding(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            PatchEmbedding,
            init_kwargs={'patch_size': 4, 'embed_dim': 2, 'normalize': False},
            input_shape=(2, 12, 12, 3),
            input_dtype='float32',
            expected_output_shape=(2, 3, 3, 2),
            expected_output_dtype='float32'
        )
        self.run_layer_test(
            PatchEmbedding,
            init_kwargs={'patch_size': 3, 'embed_dim': 2, 'normalize': True},
            input_shape=(2, 12, 12, 3),
            input_dtype='float32',
            expected_output_shape=(2, 4, 4, 2),
            expected_output_dtype='float32'
        )
