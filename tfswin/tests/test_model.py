import numpy as np
import tensorflow as tf
from keras import keras_parameterized, testing_utils, layers, models
from keras.preprocessing import image
from keras.utils import data_utils
from ..model import SwinTransformerTiny224
from ..prep import preprocess_input
from ..weight import main


def _get_elephant(target_size):
    # For models that don't include a Flatten step,
    # the default is to accept variable-size inputs
    # even when loading ImageNet weights (since it is possible).
    # In this case, default to 299x299.
    if target_size is None:
        target_size = (299, 299)
    test_image = data_utils.get_file('elephant.jpg',
                                     'https://storage.googleapis.com/tensorflow/keras-applications/tests/elephant.jpg')
    img = image.load_img(test_image, target_size=tuple(target_size), interpolation='bicubic')
    x = image.img_to_array(img)
    return x[None, ...]


# @keras_parameterized.run_all_keras_modes
class TestModel(keras_parameterized.TestCase):
    def test_layer(self):
        dbg_inp = np.load('/Users/alex/Downloads/Swin-Transformer-main/dbg_0_inp.npy')
        dbg_inp = np.transpose(dbg_inp, [0, 2, 3, 1])
        e = _get_elephant((224, 224))
        e = preprocess_input(e)
        self.assertTrue(np.all(np.abs(dbg_inp - e) < 1e-6))

        m = SwinTransformerTiny224(weights=None)
        main(m)

#         dbg_mod = models.Model(inputs=m.inputs, outputs=m.get_layer('patch_embed').output)
#         dbg_res = dbg_mod(e, training=False)
#         dbg_trg = np.load('/Users/alex/Downloads/Swin-Transformer-main/dbg_1_emb.npy')
#         # 2.26-e6 before norm, 2.39e-6 after
#         self.assertLess(np.abs(dbg_res - dbg_trg).max(), 2.39e-6)
#
#         dbg_mod = models.Model(inputs=m.inputs, outputs=m.get_layer('pos_drop').output)
#         dbg_res = dbg_mod(e, training=False)
#         dbg_trg = np.load('/Users/alex/Downloads/Swin-Transformer-main/dbg_2_pos.npy')
#         self.assertLess(np.abs(dbg_res - dbg_trg).max(), 2.39e-6)
#
#         out = m.get_layer('pos_drop').output
#         out, out_ = m.get_layer('layers.0').blocks[0](out)
#         dbg_mod = models.Model(inputs=m.inputs, outputs=[out, out_])
#         dbg_res, dbg_res_ = dbg_mod(e, training=False)
#         dbg_trg_ = np.load('/Users/alex/Downloads/Swin-Transformer-main/dbg_4_basic0_block0_.npy')
#         dbg_trg = np.load('/Users/alex/Downloads/Swin-Transformer-main/dbg_4_basic0_block0.npy')
#         # 4.17e-6 after first norm
#         self.assertLess(np.abs(dbg_res_ - dbg_trg_).max(), 4.18e-6)
#         self.assertLess(np.abs(dbg_res - dbg_trg).max(), 4.18e-6)
#
#         out = m.get_layer('layers.1').output
#         out, out_ = m.get_layer('layers.2').blocks[0](out)
#         out, out_ = m.get_layer('layers.2').blocks[1](out)
#         out, out_ = m.get_layer('layers.2').blocks[2](out)
#         out, out_ = m.get_layer('layers.2').blocks[3](out)
#         out, out_ = m.get_layer('layers.2').blocks[4](out)
#         out, out_ = m.get_layer('layers.2').blocks[5](out)
#         dbg_mod = models.Model(inputs=m.inputs, outputs=[out, out_])
#         dbg_res, dbg_res_ = dbg_mod(e, training=False)
#         # dbg_trg_ = np.load('/Users/alex/Downloads/Swin-Transformer-main/dbg_4_basic2_block0_.npy')
#         dbg_trg = np.load('/Users/alex/Downloads/Swin-Transformer-main/dbg_4_basic2_block5.npy')
#         # 6.556511e-06
#         # 7.6293945e-06
#         # 2.5749207e-05
#         # 4.7683716e-05
#         # 6.1035156e-05
#         # 0.00034332275
#         # self.assertLess(np.abs(dbg_res_ - dbg_trg_).max(), 4.18e-6)
#         self.assertLess(np.abs(dbg_res - dbg_trg).max(), 4.18e-6)
#         # 5.6922436e-06
#         # 5.1259995e-06
#
#         # out = m.get_layer('pos_drop').output
#         # out = m.get_layer('layers.0').blocks[0](out)
#         # dbg_mod = models.Model(inputs=m.inputs, outputs=out)
#         # dbg_res = dbg_mod(e, training=False)
#         # dbg_trg = np.load('/Users/alex/Downloads/Swin-Transformer-main/dbg_4_basic0_block0.npy')
#         # self.assertLess(np.abs(dbg_res - dbg_trg).max(), 2.15e-6)
#
#         # dbg_mod = models.Model(inputs=m.inputs, outputs=m.get_layer('layers.1').output)
#         # dbg_res = dbg_mod(e, training=False)
#         # dbg_trg = np.load('/Users/alex/Downloads/Swin-Transformer-main/dbg_3_basic1.npy')
#         # self.assertLess(np.abs(dbg_res - dbg_trg).max(), 2.15e-6)
#
#
#
#         # out = m.get_layer('layers.1').output
#         # out = m.get_layer('layers.2').blocks[0](out)
#         # mdbg = models.Model(inputs=m.inputs, outputs=out)
#         # r = mdbg(e, training=False)
#         # dbg_bas_bl = np.load('/Users/alex/Downloads/Swin-Transformer-main/dbg_4_basic2_block0.npy')
#         # self.assertTrue(np.all(np.abs(dbg_bas_bl - r) < 1.001e-5))
#
#         # print('-----------')
#
#         # out = m.get_layer('layers.1').output
#         # out, _, _ = m.get_layer('layers.2').blocks[0](out)
#         # out, out0, out1 = m.get_layer('layers.2').blocks[1](out)
#         # mdbg = models.Model(inputs=m.inputs, outputs=[out0, out1, out])
#         # r0, r1, r = mdbg(e, training=False)
#         # dbg_bas_bl = np.load('/Users/alex/Downloads/Swin-Transformer-main/dbg_4_basic2_block1_dbg.npy')
#         # dbg_bas_bl0 = np.load('/Users/alex/Downloads/Swin-Transformer-main/dbg_4_basic2_block1_dbg0.npy')
#         # dbg_bas_bl1 = np.load('/Users/alex/Downloads/Swin-Transformer-main/dbg_4_basic2_block1_dbg1.npy')
#         # self.assertTrue(np.all(np.abs(dbg_bas_bl0 - r0) < 1.001e-5))
#         # self.assertTrue(np.all(np.abs(dbg_bas_bl1 - r1) < 1.001e-5))
#         # print(np.max(np.abs(dbg_bas_bl - r)))
#         # self.assertTrue(np.all(np.abs(dbg_bas_bl - r) < 1.1.001e-5))
#
#         # out = m.get_layer('layers.1').output
#         # out = m.get_layer('layers.2').blocks[0](out)
#         # out = m.get_layer('layers.2').blocks[1](out)
#         # out = m.get_layer('layers.2').blocks[2](out)
#         # mdbg = models.Model(inputs=m.inputs, outputs=out)
#         # r = mdbg(dbg_inp, training=False)
#         # dbg_bas_bl = np.load('/Users/alex/Downloads/Swin-Transformer-main/dbg_4_basic2_block2.npy')
#         # print(np.max(np.abs(dbg_bas_bl - r)))
#         # self.assertTrue(np.all(np.abs(dbg_bas_bl - r) < 1.1.001e-5))
#
#         # mdbg = models.Model(inputs=m.inputs, outputs=m.get_layer('layers.2').output)
#         # r = mdbg(e, training=False)
#         # dbg_bas = np.load('/Users/alex/Downloads/Swin-Transformer-main/dbg_3_basic2.npy')
#         # self.assertLess(np.abs(dbg_bas - r).max(), 1.1.001e-5)
#
#         # mdbg = models.Model(inputs=m.inputs, outputs=m.get_layer('head').output)
#         # r = mdbg(dbg_inp, training=False)
#         # dbg_pred = np.load('/Users/alex/Develop/tfswin/tfswin/tests/res.npy')
#         # print(np.max(np.abs(dbg_pred - r)))
#         # self.assertTrue(np.all(np.abs(dbg_pred - r) < 1.001e-5))
#
#         # mdbg = models.Model(inputs=m.inputs, outputs=m.get_layer('pred').output)
#         # r = mdbg(dbg_inp, training=False)
#         # dbg_pred = np.load('/Users/alex/Downloads/Swin-Transformer-main/dbg_99_pred.npy')
#         # # print(r[0, :4])
#         # # print(dbg_pred[0, :4])
#         # self.assertTrue(np.all(np.abs(dbg_pred - r) < 1.001e-5))
#
#
# if __name__ == '__main__':
#     tf.test.main()
