# tfswin

Keras (TensorFlow v2) reimplementation of **Swin Transformer** and **Swin Transformer V2** models.

+ Based on [Official Pytorch implementation](https://github.com/microsoft/Swin-Transformer).
+ Supports variable-shape inference for downstream tasks.
+ Contains pretrained weights converted from official ones.

## Installation

```bash
pip install tfswin
```

## Examples

Default usage (without preprocessing):

```python
from tfswin import SwinTransformerTiny224  # + 5 other variants and input preprocessing

# or 
# from tfswin import SwinTransformerV2Tiny256  # + 5 other variants and input preprocessing


model = SwinTransformerTiny224()  # by default will download imagenet[21k]-pretrained weights
model.compile(...)
model.fit(...)
```

Custom classification (with preprocessing):

```python
from tf_keras import layers, models
from tfswin import SwinTransformerTiny224

inputs = layers.Input(shape=(224, 224, 3), dtype='uint8')
outputs = SwinTransformerTiny224(include_top=False)(inputs)
outputs = layers.Dense(100, activation='softmax')(outputs)

model = models.Model(inputs=inputs, outputs=outputs)
model.compile(...)
model.fit(...)
```

## Differences

Code simplification:

- Pretrain input height and width are always equal
- Patch height and width are always equal
- All input shapes automatically evaluated (not passed through a constructor like in PyTorch)
- Downsampling have been moved out from basic layer to simplify feature extraction in downstream tasks.

Performance improvements:

- Layer normalization epsilon fixed at `1.001e-5`, inputs are casted to `float32` to use fused op implementation.
- Some layers have been refactored to use faster TF operations.
- A lot of reshapes have been removed. Most of the time internal representation is 4D-tensor.
- Attention mask and relative index estimations moved to basic layer level.

## Variable shapes

Swin Transformer receptive field is larger or equal to pretraining image size. Window reduction is used in image
classification [V1](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L206)
and [V2](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py#L228) pipelines. E.g.:

- SwinTransformerTiny224: last stage size is 7x7 with window size 7, no shift for last stage.
- SwinTransformerLarge384: last stage size is 12x12 with window size 12, no shift for last stage.
- SwinTransformerV2Tiny256: last stages sizes are 16x16 and 8x8 with window size 16 and 16->6, no shift for 2 last
  stages.
- SwinTransformerV2Large384: last stages sizes are 24x24 and 12x12 with window size 24 and 24->12, no shift for 2 last
  stages.

But there is no such trick in semantic segmentation
[backbone](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/mmseg/models/backbones/swin_transformer.py#L180).
This reimplementation always applies window reduction conditioned on dynamic input height and width.

When using Swin models with input shapes different from pretraining one, try to make height and width to be multiple
of `32 * window_size`. Otherwise a lot of tensors will be padded, resulting in speed degradation.

## Evaluation

For correctness, `Tiny` and `Small` models (original and ported) tested
with [ImageNet-v2 test set](https://www.tensorflow.org/datasets/catalog/imagenet_v2).

Note: Swin models are very sensitive to input preprocessing (bicubic resize in the original evaluation script).

```python
import tensorflow as tf
import tensorflow_datasets as tfds
from tfswin import SwinTransformerTiny224


def _prepare(example, input_size=224, crop_pct=0.875):
    scale_size = tf.math.floor(input_size / crop_pct)

    image = example['image']

    shape = tf.shape(image)[:2]
    shape = tf.cast(shape, 'float32')
    shape *= scale_size / tf.reduce_min(shape)
    shape = tf.round(shape)
    shape = tf.cast(shape, 'int32')

    image = tf.image.resize(image, shape, method=tf.image.ResizeMethod.BICUBIC)
    image = tf.round(image)
    image = tf.clip_by_value(image, 0., 255.)
    image = tf.cast(image, 'uint8')

    pad_h, pad_w = tf.unstack((shape - input_size) // 2)
    image = image[pad_h:pad_h + input_size, pad_w:pad_w + input_size]

    return image, example['label']


imagenet2 = tfds.load('imagenet_v2', split='test', shuffle_files=True)
imagenet2 = imagenet2.map(_prepare, num_parallel_calls=tf.data.AUTOTUNE)
imagenet2 = imagenet2.batch(8)

model = SwinTransformerTiny224()
model.compile('sgd', 'sparse_categorical_crossentropy', ['accuracy', 'sparse_top_k_categorical_accuracy'])
history = model.evaluate(imagenet2)

print(history)
```

|   name    | original acc@1 | ported acc@1 | original acc@5 | ported acc@5 |
|:---------:|:--------------:|:------------:|:--------------:|:------------:|
| Swin-T V1 |     67.64      |  ~~67.81~~   |     87.84      |  ~~87.87~~   |
| Swin-S V1 |     70.66      |  ~~70.80~~   |     89.34      |  ~~89.49~~   |
| Swin-T V2 |     71.69      |    72.00     |     90.04      |    90.06     |
| Swin-S V2 |     73.20      |    73.57     |     91.24      |    91.11     |

Note: Swin V1 model were evaluated with wrong preprocessing (distorted aspect ratio) and ImageNet-1K weights which were
replaced with ImageNet-21K weights in 3.0.0 release.

The most metric differences comes from input data preprocessing (decoding, interpolation).
All layers outputs have been compared with original ones.
Most of them have maximum absolute difference around `9.9e-5`.
Maximum absolute difference among all layers is `3.5e-4`.

## Citation

```
@inproceedings{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
```

```
@inproceedings{liu2021swinv2,
  title={Swin Transformer V2: Scaling Up Capacity and Resolution}, 
  author={Ze Liu and Han Hu and Yutong Lin and Zhuliang Yao and Zhenda Xie and Yixuan Wei and Jia Ning and Yue Cao and Zheng Zhang and Li Dong and Furu Wei and Baining Guo},
  booktitle={International Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```