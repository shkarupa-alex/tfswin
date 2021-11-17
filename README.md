# tfswin

Keras (TensorFlow v2) reimplementation of **Swin Transformer** model.

Based on [Official Pytorch implementation](https://github.com/microsoft/Swin-Transformer).

## Examples

Default usage:

```python
from tfswin import SwinTransformerTiny224  # + 5 other variants

model = SwinTransformerTiny224()  # by default will download imagenet[21k]-pretrained weights
model.compile(...)
model.fit(...)
```

Custom classification:

```python
from keras import layers, models
from tfswin import SwinTransformerTiny224

base_model = SwinTransformerTiny224(include_top=False)
new_outputs = layers.Dense(100, activation='softmax')(base_model.outputs)
new_model = models.Model(inputs=base_model.inputs, outputs=new_outputs)

new_model.compile(...)
new_model.fit(...)
```

## Differences

Code simplification:

- Input height and width are always equal
- Patch height and width are always equal
- All input shapes automatically evaluated (not passed through a constructor)

Performance improvements:

- Layer normalization epsilon fixed at `1.001e-5`, inputs are casted to `float32` to use fused op implementation.
- Some layers (like PatchMerging) have been refactored to use faster TF operations.

## Citation

```
@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
