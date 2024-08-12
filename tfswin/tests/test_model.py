import numpy as np
from absl.testing import parameterized
from keras.src.dtype_policies import dtype_policy
from keras.src import testing
from tfswin import SwinTransformerTiny224, SwinTransformerV2Tiny256


class TestModelV1(testing.TestCase, parameterized.TestCase):
    def setUp(self):
        super(TestModelV1, self).setUp()
        self.default_policy = dtype_policy.dtype_policy()

    def tearDown(self):
        super(TestModelV1, self).tearDown()
        dtype_policy.set_dtype_policy(self.default_policy)

    @parameterized.parameters((False,), (True,))
    def test_train(self, use_fp16):
        if use_fp16:
            dtype_policy.set_dtype_policy('mixed_float16')

        model = SwinTransformerTiny224(weights=None)
        model.compile(optimizer='rmsprop', loss='mse')

        images = np.random.random((10, 224, 224, 3)).astype('float32')
        labels = (np.random.random((10, 1)) + 0.5).astype('int32')
        model.fit(images, labels, epochs=1, batch_size=2)

        # test config
        model.get_config()

    def test_var_shape(self):
        model = SwinTransformerTiny224(weights=None, include_top=False, input_shape=(None, None, 3))
        model.compile(optimizer='rmsprop', loss='mse')

        images = np.random.random((10, 512, 384, 3)).astype('float32')
        labels = (np.random.random((10, 16, 12, 768)) + 0.5).astype('int32')
        model.fit(images, labels, epochs=1, batch_size=2)

        # test config
        model.get_config()


class TestModelV2(testing.TestCase, parameterized.TestCase):
    def setUp(self):
        super(TestModelV2, self).setUp()
        self.default_policy = dtype_policy.dtype_policy()

    def tearDown(self):
        super(TestModelV2, self).tearDown()
        dtype_policy.set_dtype_policy(self.default_policy)

    @parameterized.parameters((False,), (True,))
    def test_train(self, use_fp16):
        if use_fp16:
            dtype_policy.set_dtype_policy('mixed_float16')

        model = SwinTransformerV2Tiny256(weights=None)
        model.compile(optimizer='rmsprop', loss='mse')

        images = np.random.random((10, 256, 256, 3)).astype('float32')
        labels = (np.random.random((10, 1)) + 0.5).astype('int32')
        model.fit(images, labels, epochs=1, batch_size=2)

        # test config
        model.get_config()

    def test_var_shape(self):
        model = SwinTransformerV2Tiny256(weights=None, include_top=False, input_shape=(None, None, 3))
        model.compile(optimizer='rmsprop', loss='mse')

        images = np.random.random((10, 512, 384, 3)).astype('float32')
        labels = (np.random.random((10, 16, 12, 768)) + 0.5).astype('int32')
        model.fit(images, labels, epochs=1, batch_size=2)

        # test config
        model.get_config()
