from keras.src import testing
from tfswin.mlp import MLP


class TestMLP(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            MLP,
            init_kwargs={'ratio': 0.5, 'dropout': 0.},
            input_shape=(2, 4, 4, 3),
            input_dtype='float32',
            expected_output_shape=(2, 4, 4, 3),
            expected_output_dtype='float32'
        )
        self.run_layer_test(
            MLP,
            init_kwargs={'ratio': 1.5, 'dropout': 0.2},
            input_shape=(2, 4, 4, 3),
            input_dtype='float32',
            expected_output_shape=(2, 4, 4, 3),
            expected_output_dtype='float32'
        )
