import numpy as np
import os
import tensorflow as tf
from keras import keras_parameterized, testing_utils
from tfswin.mlp import MLP


@keras_parameterized.run_all_keras_modes
class TestMLP(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            MLP,
            kwargs={'ratio': 0.5, 'dropout': 0.},
            input_shape=[2, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 3],
            expected_output_dtype='float32'
        )
        testing_utils.layer_test(
            MLP,
            kwargs={'ratio': 1.5, 'dropout': 0.2},
            input_shape=[2, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 3],
            expected_output_dtype='float32'
        )

    def test_value(self):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        inputs = np.load(f'{data_dir}/_mlp_input.npy')
        targets = np.load(f'{data_dir}/_mlp_output.npy')
        layer = MLP(4., 0.)
        layer(inputs)  # build
        layer.set_weights([
            np.load(f'{data_dir}/_mlp_fc1_weight.npy').T,
            np.load(f'{data_dir}/_mlp_fc1_bias.npy').T,
            np.load(f'{data_dir}/_mlp_fc2_weight.npy').T,
            np.load(f'{data_dir}/_mlp_fc2_bias.npy').T,
        ])
        outputs = self.evaluate(layer(inputs))
        self.assertLess(np.abs(targets - outputs).max(), 2.29e-5)


if __name__ == '__main__':
    tf.test.main()
