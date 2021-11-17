import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from keras import keras_parameterized, models, testing_utils
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

        model = SwinTransformerTiny224(weights=None)  # TODO
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

    # def test_layer(self):
    #     dbg_inp = np.load('/Users/alex/Downloads/Swin-Transformer-main/dbg_0_inp.npy')
    #     dbg_inp = np.transpose(dbg_inp, [0, 2, 3, 1])
    #
    #     m = model.SwinTransformerTiny224(weights='/Users/alex/Develop/tfswin/weights/swin_tiny_patch4_window7_224.h5')
    #
    #     dbg_mod = models.Model(inputs=m.inputs, outputs=m.get_layer('patch_embed').output)
    #     dbg_res = dbg_mod(e, training=False)
    #     dbg_trg = np.load('/Users/alex/Downloads/Swin-Transformer-main/dbg_1_emb.npy')
    #     # 2.26-e6 before norm, 2.39e-6 after
    #     self.assertLess(np.abs(dbg_res - dbg_trg).max(), 2.39e-6)
    #
    #     dbg_mod = models.Model(inputs=m.inputs, outputs=m.get_layer('pos_drop').output)
    #     dbg_res = dbg_mod(e, training=False)
    #     dbg_trg = np.load('/Users/alex/Downloads/Swin-Transformer-main/dbg_2_pos.npy')
    #     self.assertLess(np.abs(dbg_res - dbg_trg).max(), 2.39e-6)
    #
    #     out = m.get_layer('pos_drop').output
    #     out = m.get_layer('layers.0').blocks[0](out)
    #     dbg_mod = models.Model(inputs=m.inputs, outputs=[out])
    #     dbg_res = dbg_mod(e, training=False)
    #     dbg_trg = np.load('/Users/alex/Downloads/Swin-Transformer-main/dbg_4_basic0_block0.npy')
    #     self.assertLess(np.abs(dbg_res - dbg_trg).max(), 4.18e-6)
    #
    #     #         out = m.get_layer('layers.1').output
    #     #         out, out_ = m.get_layer('layers.2').blocks[0](out)
    #     #         out, out_ = m.get_layer('layers.2').blocks[1](out)
    #     #         out, out_ = m.get_layer('layers.2').blocks[2](out)
    #     #         out, out_ = m.get_layer('layers.2').blocks[3](out)
    #     #         out, out_ = m.get_layer('layers.2').blocks[4](out)
    #     #         out, out_ = m.get_layer('layers.2').blocks[5](out)
    #     #         dbg_mod = models.Model(inputs=m.inputs, outputs=[out, out_])
    #     #         dbg_res, dbg_res_ = dbg_mod(e, training=False)
    #     #         # dbg_trg_ = np.load('/Users/alex/Downloads/Swin-Transformer-main/dbg_4_basic2_block0_.npy')
    #     #         dbg_trg = np.load('/Users/alex/Downloads/Swin-Transformer-main/dbg_4_basic2_block5.npy')
    #     #         # 6.556511e-06
    #     #         # 7.6293945e-06
    #     #         # 2.5749207e-05
    #     #         # 4.7683716e-05
    #     #         # 6.1035156e-05
    #     #         # 0.00034332275
    #     #         # self.assertLess(np.abs(dbg_res_ - dbg_trg_).max(), 4.18e-6)
    #     #         self.assertLess(np.abs(dbg_res - dbg_trg).max(), 4.18e-6)
    #     #         # 5.6922436e-06
    #     #         # 5.1259995e-06
    #     #
    #     #         # out = m.get_layer('pos_drop').output
    #     #         # out = m.get_layer('layers.0').blocks[0](out)
    #     #         # dbg_mod = models.Model(inputs=m.inputs, outputs=out)
    #     #         # dbg_res = dbg_mod(e, training=False)
    #     #         # dbg_trg = np.load('/Users/alex/Downloads/Swin-Transformer-main/dbg_4_basic0_block0.npy')
    #     #         # self.assertLess(np.abs(dbg_res - dbg_trg).max(), 2.15e-6)
    #
    #     # dbg_mod = models.Model(inputs=m.inputs, outputs=m.get_layer('layers.1').output)
    #     # dbg_res = dbg_mod(e, training=False)
    #     # dbg_trg = np.load('/Users/alex/Downloads/Swin-Transformer-main/dbg_3_basic1.npy')
    #     # self.assertLess(np.abs(dbg_res - dbg_trg).max(), 2.15e-6)
    #
    #     dbg_mod = models.Model(inputs=m.inputs, outputs=m.get_layer('layers.3').output)
    #     dbg_res = dbg_mod(e, training=False)
    #     dbg_trg = np.load('/Users/alex/Downloads/Swin-Transformer-main/dbg_3_basic3.npy')
    #     self.assertLess(np.abs(dbg_res - dbg_trg).max(), 2.15e-6)


if __name__ == '__main__':
    tf.test.main()
