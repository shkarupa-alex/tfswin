import tensorflow as tf
from keras.src import testing
from tfswin.ape import AbsoluteEmbedding


class TestAbsoluteEmbedding(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            AbsoluteEmbedding,
            init_kwargs={'pretrain_size': 56},
            input_shape=(2, 56, 56, 3),
            input_dtype='float32',
            expected_output_shape=(2, 56, 56, 3),
            expected_output_dtype='float32'
        )
        self.run_layer_test(
            AbsoluteEmbedding,
            init_kwargs={'pretrain_size': 56},
            input_shape=(2, 112, 112, 3),
            input_dtype='float32',
            expected_output_shape=(2, 112, 112, 3),
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
