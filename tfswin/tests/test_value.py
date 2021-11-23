import os.path

import tensorflow as tf
import numpy as np
import tfswin
from scipy.spatial.distance import cosine as cosine_distance


class TestValue(tf.test.TestCase):
    # def test_embed(self):
    #     inputs = np.load('/Users/alex/Downloads/Swin-Transformer-main/dbg/embed_0_inp.npy').transpose([0, 2, 3, 1])
    #     middles = np.load('/Users/alex/Downloads/Swin-Transformer-main/dbg/embed_0_mid.npy')
    #     outputs = np.load('/Users/alex/Downloads/Swin-Transformer-main/dbg/embed_0_out.npy')
    #
    #     model = tfswin.SwinTransformerTiny224()
    #     layer = model.get_layer('patch_embed')
    #
    #     results = self.evaluate(layer.proj(inputs))
    #     results = results.reshape([inputs.shape[0], -1, 96])
    #     self.assertLess(np.abs(middles - results).max(), 9.6e-7)
    #     self.assertLess(cosine_distance(middles.ravel(), results.ravel()), 1.2e-7)
    #
    #     results = self.evaluate(layer(inputs))
    #     self.assertLess(np.abs(outputs - results).max(), 2.9e-6)
    #     self.assertLess(cosine_distance(outputs.ravel(), results.ravel()), 1.2e-7)

    # def test_attn(self):
    #     model = tfswin.SwinTransformerTiny224()
    #     model.compile()
    #
    #     layers = [l for l in model._flatten_layers() if 'attn' in l.name]
    #     assert 12 == len(layers)
    #
    #     heads_ = [3, 3, 6, 6, 12, 12, 12, 12, 12, 12, 24, 24]
    #     masks_ = [False, True, False, True, False, True, False, True, False, True, False, False]
    #
    #     values = []
    #     values_cosine = []
    #     for i in range(12):
    #         inputs = np.load(f'/Users/alex/Downloads/Swin-Transformer-main/dbg/winatt_{i}_inp.npy')
    #         masks = None
    #         if masks_[i]:
    #             masks = np.load(f'/Users/alex/Downloads/Swin-Transformer-main/dbg/winatt_{i}_mid.npy')
    #         outputs = np.load(f'/Users/alex/Downloads/Swin-Transformer-main/dbg/winatt_{i}_out.npy')
    #
    #         self.assertEqual(7, layers[i].get_config()['window_size'])
    #         self.assertEqual(heads_[i], layers[i].get_config()['num_heads'])
    #         self.assertEqual(True, layers[i].get_config()['qkv_bias'])
    #         self.assertEqual(None, layers[i].get_config()['qk_scale'])
    #         self.assertEqual(0., layers[i].get_config()['attn_drop'])
    #         self.assertEqual(0., layers[i].get_config()['proj_drop'])
    #         self.assertEqual(masks_[i], layers[i].get_config()['attn_mask'])
    #
    #         if masks_[i]:
    #             results = self.evaluate(layers[i]([inputs, masks]))
    #         else:
    #             results = self.evaluate(layers[i](inputs))
    #         values.append(np.abs(outputs - results).max())
    #         values_cosine.append(cosine_distance(outputs.ravel(), results.ravel()))
    #
    #     print(min(values), max(values))
    #     self.assertLess(max(values), 3.1e-5)
    #
    #     print(min(values_cosine), max(values_cosine))
    #     self.assertLess(max(values_cosine), 1.2e-7)

    # def test_merge(self):
    #     model = tfswin.SwinTransformerTiny224()
    #     model.compile()
    #
    #     layers = [l for l in model._flatten_layers() if 'downsample' in l.name]
    #     assert 3 == len(layers)
    #
    #     values = []
    #     values_cosine = []
    #     for i in range(3):
    #         inputs = np.load(f'/Users/alex/Downloads/Swin-Transformer-main/dbg/merge_{i}_inp.npy')
    #         outputs = np.load(f'/Users/alex/Downloads/Swin-Transformer-main/dbg/merge_{i}_out.npy')
    #         results = self.evaluate(layers[i](inputs))
    #         values.append(np.abs(outputs - results).max())
    #         values_cosine.append(cosine_distance(outputs.ravel(), results.ravel()))
    #
    #     print(min(values), max(values))
    #     self.assertLess(max(values), 6.9e-5)
    #
    #     print(min(values_cosine), max(values_cosine))
    #     self.assertLess(max(values_cosine), 1.2e-7)

    # def test_mlp(self):
    #     model = tfswin.SwinTransformerTiny224()
    #     model.compile()
    #
    #     layers = [l for l in model._flatten_layers() if 'mlp' in l.name]
    #     assert 12 == len(layers)
    #
    #     values = []
    #     values_cosine = []
    #     for i in range(12):
    #         inputs = np.load(f'/Users/alex/Downloads/Swin-Transformer-main/dbg/mlp_{i}_inp.npy')
    #         outputs = np.load(f'/Users/alex/Downloads/Swin-Transformer-main/dbg/mlp_{i}_out.npy')
    #
    #         print(i, inputs.shape, outputs.shape, layers[i].input_spec)
    #
    #         self.assertEqual(4., layers[i].get_config()['ratio'])
    #         self.assertEqual(0., layers[i].get_config()['dropout'])
    #
    #         results = self.evaluate(layers[i](inputs))
    #         values.append(np.abs(outputs - results).max())
    #         values_cosine.append(cosine_distance(outputs.ravel(), results.ravel()))
    #
    #     print(min(values), max(values))
    #     self.assertLess(max(values), 6.2e-5)
    #
    #     print(min(values_cosine), max(values_cosine))
    #     self.assertLess(max(values_cosine), 1.2e-7)

    # def test_swin(self):
    #     model = tfswin.SwinTransformerTiny224()
    #     model.compile()
    #
    #     layers = [l for l in model._flatten_layers() if 'blocks.' in l.name]
    #     assert 12 == len(layers)
    #
    #     heads_ = [3, 3, 6, 6, 12, 12, 12, 12, 12, 12, 24, 24]
    #     shifts_ = [0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3]
    #     paths_ = [0.0, 0.0181818176060915, 0.036363635212183, 0.05454545468091965, 0.072727270424366, 0.09090908616781235, 0.10909091681241989, 0.12727272510528564, 0.1454545557498932, 0.16363637149333954, 0.1818181872367859, 0.20000000298023224]
    #
    #     values = []
    #     values_cosine = []
    #     for i in range(12):
    #         inputs = np.load(f'/Users/alex/Downloads/Swin-Transformer-main/dbg/swin_{i}_inp.npy')
    #         outputs = np.load(f'/Users/alex/Downloads/Swin-Transformer-main/dbg/swin_{i}_out.npy')
    #         middles1 = np.load(f'/Users/alex/Downloads/Swin-Transformer-main/dbg/swin_{i}_mid1.npy')
    #         middles2 = np.load(f'/Users/alex/Downloads/Swin-Transformer-main/dbg/swin_{i}_mid2.npy')
    #         middles3 = np.load(f'/Users/alex/Downloads/Swin-Transformer-main/dbg/swin_{i}_mid3.npy')
    #         middles4 = np.load(f'/Users/alex/Downloads/Swin-Transformer-main/dbg/swin_{i}_mid4.npy')
    #         if os.path.exists(f'/Users/alex/Downloads/Swin-Transformer-main/dbg/swin_{i}_mid5.npy'):
    #             middles5 = np.load(f'/Users/alex/Downloads/Swin-Transformer-main/dbg/swin_{i}_mid5.npy')
    #         else:
    #             middles5 = None
    #         middles6 = np.load(f'/Users/alex/Downloads/Swin-Transformer-main/dbg/swin_{i}_mid6.npy')
    #         middles7 = np.load(f'/Users/alex/Downloads/Swin-Transformer-main/dbg/swin_{i}_mid7.npy')
    #         middles8 = np.load(f'/Users/alex/Downloads/Swin-Transformer-main/dbg/swin_{i}_mid8.npy')
    #         middles9 = np.load(f'/Users/alex/Downloads/Swin-Transformer-main/dbg/swin_{i}_mid9.npy')
    #         middles10 = np.load(f'/Users/alex/Downloads/Swin-Transformer-main/dbg/swin_{i}_mid10.npy')
    #         middles11 = np.load(f'/Users/alex/Downloads/Swin-Transformer-main/dbg/swin_{i}_mid11.npy')
    #
    #         self.assertEqual(heads_[i], layers[i].get_config()['num_heads'])
    #         self.assertEqual(7, layers[i].get_config()['window_size'])
    #         self.assertEqual(shifts_[i], layers[i].get_config()['shift_size'])
    #         self.assertEqual(4., layers[i].get_config()['mlp_ratio'])
    #         self.assertEqual(True, layers[i].get_config()['qkv_bias'])
    #         self.assertEqual(None, layers[i].get_config()['qk_scale'])
    #
    #         self.assertEqual(0., layers[i].get_config()['drop'])
    #         self.assertEqual(0., layers[i].get_config()['attn_drop'])
    #         self.assertAlmostEqual(paths_[i], layers[i].get_config()['path_drop'], places=6)
    #
    #         results = self.evaluate(layers[i](inputs))
    #         results, dbg1, dbg2, dbg3, dbg4, dbg5, dbg6, dbg7, dbg8, dbg9, dbg10, dbg11 = results
    #         values.append(np.abs(outputs - results).max())
    #         values_cosine.append(cosine_distance(outputs.ravel(), results.ravel()))
    #
    #         diff = (
    #             np.abs(middles1 - dbg1).max(),
    #             np.abs(middles2 - dbg2).max(),
    #             np.abs(middles3 - dbg3).max(),
    #             np.abs(middles4 - dbg4).max(),
    #             0. if middles5 is None else np.abs(middles5 - dbg5).max(),
    #             np.abs(middles6 - dbg6).max(),
    #             np.abs(middles7 - dbg7).max(),
    #             np.abs(middles8 - dbg8).max(),
    #             np.abs(middles9 - dbg9).max(),
    #             np.abs(middles10 - dbg10).max(),
    #             np.abs(middles11 - dbg11).max(),
    #         )
    #         print(i, max(diff))
    #         if max(diff) > 1e-4:
    #             print(diff)
    #
    #     print(min(values), max(values))
    #     self.assertLess(max(values), 1.5e-4)
    #
    #     print(min(values_cosine), max(values_cosine))
    #     self.assertLess(max(values_cosine), 1.2e-7)

    # def test_basic(self):
    #     model = tfswin.SwinTransformerTiny224()
    #     model.compile()
    #
    #     layers = [l for l in model._flatten_layers() if 'layers.' in l.name]
    #     assert 4 == len(layers)
    #
    #     depths_ = [2, 2, 6, 2]
    #     heads_ = [3, 6, 12, 24]
    #     paths_ = [[0.0, 0.0181818176060915], [0.036363635212183, 0.05454545468091965], [0.072727270424366, 0.09090908616781235, 0.10909091681241989, 0.12727272510528564, 0.1454545557498932, 0.16363637149333954], [0.1818181872367859, 0.20000000298023224]]
    #     downs_ = [True, True, True, False]
    #
    #     values = []
    #     values_cosine = []
    #     for i in range(4):
    #         inputs = np.load(f'/Users/alex/Downloads/Swin-Transformer-main/dbg/basic_{i}_inp.npy')
    #         outputs = np.load(f'/Users/alex/Downloads/Swin-Transformer-main/dbg/basic_{i}_out.npy')
    #
    #         self.assertEqual(depths_[i], layers[i].get_config()['depth'])
    #         self.assertEqual(heads_[i], layers[i].get_config()['num_heads'])
    #         self.assertEqual(7, layers[i].get_config()['window_size'])
    #         self.assertEqual(4., layers[i].get_config()['mlp_ratio'])
    #         self.assertEqual(True, layers[i].get_config()['qkv_bias'])
    #         self.assertEqual(None, layers[i].get_config()['qk_scale'])
    #         self.assertEqual(0., layers[i].get_config()['drop'])
    #         self.assertEqual(0., layers[i].get_config()['attn_drop'])
    #
    #         for j, p in enumerate(paths_[i]):
    #             self.assertAlmostEqual(p, layers[i].get_config()['path_drop'][j], places=7)
    #
    #         self.assertEqual(downs_[i], layers[i].get_config()['downsample'])
    #
    #         results = self.evaluate(layers[i](inputs))
    #         values.append(np.abs(outputs - results).max())
    #         values_cosine.append(cosine_distance(outputs.ravel(), results.ravel()))
    #
    #     print(min(values), max(values))
    #     self.assertLess(max(values), 3.5e-4)
    #
    #     print(min(values_cosine), max(values_cosine))
    #     self.assertLess(max(values_cosine), 1.2e-7)

    def test_model(self):
        inputs = np.load('/Users/alex/Downloads/Swin-Transformer-main/dbg/model_inp.npy').transpose([0, 2, 3, 1])
        outputs = np.load('/Users/alex/Downloads/Swin-Transformer-main/dbg/model_out.npy')

        model = tfswin.SwinTransformerTiny224(classifier_activation=None)
        model.compile()

        results = model.predict(inputs)
        self.assertTupleEqual(outputs.shape, results.shape)
        self.assertLess(np.abs(outputs - results).max(), 1.7e-5)
        self.assertLess(cosine_distance(outputs.ravel(), results.ravel()), 1.2e-7)


if __name__ == '__main__':
    tf.test.main()
