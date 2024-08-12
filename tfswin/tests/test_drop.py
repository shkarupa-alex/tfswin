from keras.src import testing
from tfswin.drop import DropPath


class TestDropPath(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            DropPath,
            init_kwargs={'rate': 0.},
            input_shape=(2, 16, 3),
            input_dtype='float32',
            expected_output_shape=(2, 16, 3),
            expected_output_dtype='float32'
        )
        self.run_layer_test(
            DropPath,
            init_kwargs={'rate': 0.2},
            input_shape=(2, 16, 16, 3),
            input_dtype='float32',
            expected_output_shape=(2, 16, 16, 3),
            expected_output_dtype='float32'
        )
