# tfswin

Keras (TFv2) reimplementation of **Swin Transformer** model.

Based on [Official Pytorch implementation](https://github.com/microsoft/Swin-Transformer).

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

## TODO:
init