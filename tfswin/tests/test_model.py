import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from keras import keras_parameterized, testing_utils
from keras.mixed_precision import policy as mixed_precision
from tfswin import SwinTransformerTiny224
from tensorflow.python.util import object_identity
from tensorflow.python.training.tracking import util as trackable_util


@keras_parameterized.run_all_keras_modes
class TestModel(keras_parameterized.TestCase):
    def setUp(self):
        super(TestModel, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestModel, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    @parameterized.parameters((False,), (True,))
    def test_train(self, use_fp16):
        if use_fp16:
            mixed_precision.set_global_policy('mixed_float16')

        model = SwinTransformerTiny224(weights=None)
        model.compile(optimizer='rmsprop', loss='mse', run_eagerly=testing_utils.should_run_eagerly())

        images = np.random.random((10, 224, 224, 3)).astype('float32')
        labels = (np.random.random((10, 1)) + 0.5).astype('int32')
        model.fit(images, labels, epochs=1, batch_size=2)

        # test config
        model.get_config()

        # check whether the model variables are present in the trackable list of objects
        checkpointed_objects = object_identity.ObjectIdentitySet(trackable_util.list_objects(model))
        for v in model.variables:
            self.assertIn(v, checkpointed_objects)


if __name__ == '__main__':
    tf.test.main()
