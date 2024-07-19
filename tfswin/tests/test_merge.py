import tensorflow as tf
from keras.src import testing
from tfswin.merge import PatchMerging


class TestPatchMerging(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            PatchMerging,
            init_kwargs={'swin_v2': False},
            input_shape=(2, 12, 12, 4),
            input_dtype='float32',
            expected_output_shape=(2, 6, 6, 8),
            expected_output_dtype='float32'
        )

        self.run_layer_test(
            PatchMerging,
            init_kwargs={'swin_v2': True},
            input_shape=(2, 11, 13, 3),
            input_dtype='float32',
            expected_output_shape=(2, 6, 7, 6),
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
