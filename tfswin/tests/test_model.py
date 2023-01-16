import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from keras.testing_infra import test_combinations, test_utils
from keras.mixed_precision import policy as mixed_precision
from tfswin import SwinTransformerTiny224, SwinTransformerV2Tiny256
from tensorflow.python.util import object_identity
from tensorflow.python.training.tracking import util as trackable_util


@test_combinations.run_all_keras_modes
class TestModelV1(test_combinations.TestCase):
    def setUp(self):
        super(TestModelV1, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestModelV1, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    @parameterized.parameters((False,), (True,))
    def test_train(self, use_fp16):
        if use_fp16:
            mixed_precision.set_global_policy('mixed_float16')

        model = SwinTransformerTiny224(weights=None)
        model.compile(optimizer='rmsprop', loss='mse', run_eagerly=test_utils.should_run_eagerly())

        images = np.random.random((10, 224, 224, 3)).astype('float32')
        labels = (np.random.random((10, 1)) + 0.5).astype('int32')
        model.fit(images, labels, epochs=1, batch_size=2)

        # test config
        model.get_config()

        # check whether the model variables are present in the trackable list of objects
        checkpointed_objects = object_identity.ObjectIdentitySet(trackable_util.list_objects(model))
        for v in model.variables:
            self.assertIn(v, checkpointed_objects)

    def test_ape(self):
        model = SwinTransformerTiny224(weights=None, use_ape=True)
        model.compile(optimizer='rmsprop', loss='mse', run_eagerly=test_utils.should_run_eagerly())

        images = np.random.random((10, 224, 224, 3)).astype('float32')
        labels = (np.random.random((10, 1)) + 0.5).astype('int32')
        model.fit(images, labels, epochs=1, batch_size=2)

        # test config
        model.get_config()

        # check whether the model variables are present in the trackable list of objects
        checkpointed_objects = object_identity.ObjectIdentitySet(trackable_util.list_objects(model))
        for v in model.variables:
            self.assertIn(v, checkpointed_objects)

    def test_var_shape(self):
        model = SwinTransformerTiny224(weights=None, include_top=False, input_shape=(None, None, 3))
        run_eagerly = test_utils.should_run_eagerly()
        model.compile(optimizer='rmsprop', loss='mse', run_eagerly=run_eagerly, jit_compile=not run_eagerly)

        images = np.random.random((10, 512, 384, 3)).astype('float32')
        labels = (np.random.random((10, 16, 12, 768)) + 0.5).astype('int32')
        model.fit(images, labels, epochs=1, batch_size=2)

        # test config
        model.get_config()

        # check whether the model variables are present in the trackable list of objects
        checkpointed_objects = object_identity.ObjectIdentitySet(trackable_util.list_objects(model))
        for v in model.variables:
            self.assertIn(v, checkpointed_objects)


@test_combinations.run_all_keras_modes
class TestModelV2(test_combinations.TestCase):
    def setUp(self):
        super(TestModelV2, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestModelV2, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    @parameterized.parameters((False,), (True,))
    def test_train(self, use_fp16):
        if use_fp16:
            mixed_precision.set_global_policy('mixed_float16')

        model = SwinTransformerV2Tiny256(weights=None)
        model.compile(optimizer='rmsprop', loss='mse', run_eagerly=test_utils.should_run_eagerly())

        images = np.random.random((10, 256, 256, 3)).astype('float32')
        labels = (np.random.random((10, 1)) + 0.5).astype('int32')
        model.fit(images, labels, epochs=1, batch_size=2)

        # test config
        model.get_config()

        # check whether the model variables are present in the trackable list of objects
        checkpointed_objects = object_identity.ObjectIdentitySet(trackable_util.list_objects(model))
        for v in model.variables:
            self.assertIn(v, checkpointed_objects)

    def test_var_shape(self):
        model = SwinTransformerV2Tiny256(weights=None, include_top=False, input_shape=(None, None, 3))
        run_eagerly = test_utils.should_run_eagerly()
        model.compile(optimizer='rmsprop', loss='mse', run_eagerly=run_eagerly, jit_compile=not run_eagerly)

        images = np.random.random((10, 512, 384, 3)).astype('float32')
        labels = (np.random.random((10, 16, 12, 768)) + 0.5).astype('int32')
        model.fit(images, labels, epochs=1, batch_size=2)

        # test config
        model.get_config()

        # check whether the model variables are present in the trackable list of objects
        checkpointed_objects = object_identity.ObjectIdentitySet(trackable_util.list_objects(model))
        for v in model.variables:
            self.assertIn(v, checkpointed_objects)


if __name__ == '__main__':
    tf.test.main()
