# tfswin

Keras (TensorFlow v2) reimplementation of **Swin Transformer** model.

Based on [Official Pytorch implementation](https://github.com/microsoft/Swin-Transformer).

Supports variable-shape inference.

## Examples

Default usage (without preprocessing):

```python
from tfswin import SwinTransformerTiny224  # + 5 other variants and input preprocessing

model = SwinTransformerTiny224()  # by default will download imagenet[21k]-pretrained weights
model.compile(...)
model.fit(...)
```

Custom classification (with preprocessing):

```python
from keras import layers, models
from tfswin import SwinTransformerTiny224, preprocess_input

inputs = layers.Input(shape=(224, 224, 3), dtype='uint8')
outputs = layers.Lambda(preprocess_input)(inputs)
outputs = SwinTransformerTiny224(include_top=False)(outputs)
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
- Attention mask estimation moved to basic layer level.

## Variable shapes

When using Swin models with shapes different from pretraining one, try to make height and width to be multiple
of `32 * window_size`. Otherwise a lot of tensors will be padded, resulting in speed and (possibly) quality degradation.

## Evaluation

For correctness, `Tiny` and `Small` models (original and ported) tested
with [ImageNet-v2 test set](https://www.tensorflow.org/datasets/catalog/imagenet_v2).

Note, swin models are very sensitive to input preprocessing (bicubic resize with antialiasing in the original evaluation
script).

```python
import tensorflow as tf
import tensorflow_datasets as tfds
from tfswin import SwinTransformerTiny224, preprocess_input


def _prepare(example):
    image = tf.image.resize(example['image'], (224, 224), method=tf.image.ResizeMethod.BICUBIC, antialias=True)
    image = preprocess_input(image)

    return image, example['label']


imagenet2 = tfds.load('imagenet_v2', split='test', shuffle_files=True)
imagenet2 = imagenet2.map(_prepare, num_parallel_calls=tf.data.AUTOTUNE)
imagenet2 = imagenet2.batch(8)

model = SwinTransformerTiny224()
model.compile('sgd', 'sparse_categorical_crossentropy', ['accuracy', 'sparse_top_k_categorical_accuracy'])
history = model.evaluate(imagenet2)

print(history)
```

| name | original acc@1 | ported acc@1 | original acc@5 | ported acc@5 |
| :---: | :---: | :---: | :---: | :---: |
| Swin-T | 67.64 | 67.81 | 87.84 | 87.87 |
| Swin-S | 70.66 | 70.80 | 89.34 | 89.49 |

Meanwhile, all layers outputs have been compared with original. Most of them have maximum absolute difference
around `9.9e-5`. Maximum absolute difference among all layers is `3.5e-4`.

## Citation

```
@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
