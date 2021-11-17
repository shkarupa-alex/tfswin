import numpy as np
import tensorflow as tf
from keras import backend, layers, models
from keras.applications import imagenet_utils
from keras.utils import data_utils, layer_utils
from tfswin.ape import AbsoluteEmbedding
from tfswin.basic import BasicLayer
from tfswin.embed import PatchEmbedding
from tfswin.norm import LayerNorm

BASE_URL = 'https://github.com/shkarupa-alex/tfswin/releases/download/1.0.0/swin_{}.h5'
WEIGHT_URLS = {
    'swin_tiny_224': BASE_URL.format('tiny_patch4_window7_224'),
    'swin_small_224': BASE_URL.format('small_patch4_window7_224'),
    'swin_base_224': BASE_URL.format('base_patch4_window7_224_22k'),
    'swin_base_384': BASE_URL.format('base_patch4_window12_384_22k'),
    'swin_large_224': BASE_URL.format('large_patch4_window7_224_22k'),
    'swin_large_384': BASE_URL.format('large_patch4_window12_384_22k')
}
WEIGHT_HASHES = {
    'swin_tiny_224': 'd4978388fed2c48d3e141ff8bfadfd402626d6e405d5a89aa146cd2dd6bef04f',
    'swin_small_224': 'd4dc852d32c00762f0ba5c6b33d3a08670c3aa3377c5d10104d505260070967e',
    'swin_base_224': 'b68200eb707302e471242da86fd5a9d3b84d0c9e30698bd193b775207b5100eb',
    'swin_base_384': '48dd1a39217ac26a7c88ebe1212714812f15db6ca97ec09bed0ac85cbdf930f1',
    'swin_large_224': '8910d7390d275a06a4c1a65ab458b089f72c431351ccfec94c296f18cbcdd714',
    'swin_large_384': 'a04eb046fd5412a1f4339478e625b8f691d9811340a98b7675295fb7d1480c77'
}


def SwinTransformer(
        model_name, input_shape, window_size, embed_dim, depths, num_heads, patch_size=4, patch_norm=True,
        use_ape=False, drop_rate=0., mlp_ratio=4., qkv_bias=True, qk_scale=None, attn_drop=0., path_drop=0.1,
        include_top=True, classes=1000, classifier_activation='softmax', pooling=None, weights=None, input_tensor=None):
    """Instantiates the Swin Transformer architecture.

    Args:
      model_name: model name.
      input_shape: fully-defined shape tuple of an input image. Height and width should be equal.
      window_size: window partition size.
      embed_dim: patch embedding dimension.
      depths: depth of each Swin Transformer layer.
      num_heads: number of attention heads.
      patch_size: patch size used to divide input image.
      patch_norm: whether to add normalization after patch embedding.
      use_ape: whether to add absolute position embedding to the patch embedding
      drop_rate: dropout rate.
      mlp_ratio: ratio of mlp hidden units to embedding units.
      qkv_bias: whether to add a learnable bias to query, key, value.
      qk_scale: override default qk scale of head_dim ** -0.5 if set
      attn_drop: attention dropout rate
      path_drop: stochastic depth rate
      include_top: whether to include the fully-connected layer at the top of the network.
      classes: optional number of classes to classify images into, only to be specified if `include_top` is True.
      classifier_activation: the activation function to use on the "top" layer. Ignored unless `include_top=True`.
        When loading pretrained weights, `classifier_activation` can only be `None` or `"softmax"`.
      pooling: optional pooling mode for feature extraction when `include_top` is `False`.
        - `None` means that the output of the model will be the 3D tensor output of the last layer.
        - `avg` means that global average pooling will be applied to the output of the last layer, and thus the output
          of the model will be a 2D tensor.
        - `max` means that global max pooling will be applied.
      weights: one of `None` (random initialization), 'imagenet' (pre-training on ImageNet or ImageNet 21k), or the
        path to the weights file to be loaded.
      input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use as image input for the model.

    Returns:
      A `keras.Model` instance.
    """
    if not (weights in {'imagenet', None} or tf.io.gfile.exists(weights)):
        raise ValueError('The `weights` argument should be either `None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes not in {1000, 21841}:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top` as true, '
                         '`classes` should be 1000 or 21841 depending on model type')

    if input_tensor:
        try:
            backend.is_keras_tensor(input_tensor)
        except ValueError:
            raise ValueError(f'Expecting `input_tensor` to be a symbolic tensor instance. '
                             f'Got {input_tensor} of type {type(input_tensor)}')

    if input_tensor:
        tensor_shape = backend.int_shape(input_tensor)[1:]
        if input_tensor.dtype != 'float32':
            raise ValueError('DType of `input_tensor` should equals to `float32`.')

        if input_shape and tensor_shape != input_shape:
            raise ValueError('Shape of `input_tensor` should equals to `input_shape` if both provided.')
        else:
            input_shape = tensor_shape

    if input_shape[0] != input_shape[1]:
        raise ValueError('Height and width of the inputs should be equal.')

    # Determine proper input shape
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=224,
        min_size=32,
        data_format='channel_last',
        require_flatten=False,
        weights=weights)

    if input_tensor:
        if backend.is_keras_tensor(input_tensor):
            image = input_tensor
        else:
            image = layers.Input(tensor=input_tensor, shape=input_shape, dtype='float32')
    else:
        image = layers.Input(shape=input_shape)

    # Define model pipeline
    x = PatchEmbedding(patch_size=patch_size, embed_dim=embed_dim, normalize=patch_norm, name='patch_embed')(image)

    if use_ape:
        x = AbsoluteEmbedding()(x)

    x = layers.Dropout(drop_rate, name='pos_drop')(x)

    path_drops = [x for x in np.linspace(0., path_drop, sum(depths))]

    for i in range(len(depths)):
        path_drop = path_drops[sum(depths[:i]):sum(depths[:i + 1])]
        not_last = i != len(depths) - 1
        x = BasicLayer(depth=depths[i], num_heads=num_heads[i], window_size=window_size, mlp_ratio=mlp_ratio,
                       qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop,
                       path_drop=path_drop, downsample=not_last, name=f'layers.{i}')(x)

    x = LayerNorm(name='norm')(x)

    if include_top:
        x = layers.GlobalAveragePooling1D(name='avg_pool')(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Dense(classes, name='head')(x)
        x = layers.Activation(classifier_activation, dtype='float32', name='pred')(x)
    elif pooling == 'avg':
        x = layers.GlobalAveragePooling1D(name='avg_pool')(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling1D(name='max_pool')(x)

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = image

    # Create model.
    model = models.Model(inputs, x, name=model_name)

    # Load weights.
    if 'imagenet' == weights and model_name in WEIGHT_URLS:
        weights_url = WEIGHT_URLS[model_name]
        weights_hash = WEIGHT_HASHES[model_name]
        weights_path = data_utils.get_file(origin=weights_url, file_hash=weights_hash, cache_subdir='tfswin')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    if include_top:
        return model

    last_layer = 'norm'
    if pooling == 'avg':
        last_layer = 'avg_pool'
    elif pooling == 'max':
        last_layer = 'max_pool'

    outputs = model.get_layer(name=last_layer).output
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)

    return model


def SwinTransformerTiny224(model_name='swin_tiny_224', input_shape=(224, 224, 3), window_size=7, embed_dim=96,
                           depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), weights='imagenet', **kwargs):
    return SwinTransformer(model_name=model_name, input_shape=input_shape, window_size=window_size, embed_dim=embed_dim,
                           depths=depths, num_heads=num_heads, weights=weights, **kwargs)


def SwinTransformerSmall224(model_name='swin_small_224', input_shape=(224, 224, 3), window_size=7, embed_dim=96,
                            depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), weights='imagenet', **kwargs):
    return SwinTransformer(model_name=model_name, input_shape=input_shape, window_size=window_size, embed_dim=embed_dim,
                           depths=depths, num_heads=num_heads, weights=weights, **kwargs)


def SwinTransformerBase224(model_name='swin_base_224', input_shape=(224, 224, 3), window_size=7, embed_dim=128,
                           depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), classes=21841, weights='imagenet', **kwargs):
    return SwinTransformer(model_name=model_name, input_shape=input_shape, window_size=window_size, embed_dim=embed_dim,
                           depths=depths, num_heads=num_heads, weights=weights, classes=classes, **kwargs)


def SwinTransformerBase384(model_name='swin_base_384', input_shape=(384, 384, 3), window_size=12, embed_dim=128,
                           depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), classes=21841, weights='imagenet', **kwargs):
    return SwinTransformer(model_name=model_name, input_shape=input_shape, window_size=window_size, embed_dim=embed_dim,
                           depths=depths, num_heads=num_heads, weights=weights, classes=classes, **kwargs)


def SwinTransformerLarge224(model_name='swin_large_224', input_shape=(224, 224, 3), window_size=7, embed_dim=192,
                            depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48), classes=21841, weights='imagenet',
                            **kwargs):
    return SwinTransformer(model_name=model_name, input_shape=input_shape, window_size=window_size, embed_dim=embed_dim,
                           depths=depths, num_heads=num_heads, weights=weights, classes=classes, **kwargs)


def SwinTransformerLarge384(model_name='swin_large_384', input_shape=(384, 384, 3), window_size=12, embed_dim=192,
                            depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48), classes=21841, weights='imagenet',
                            **kwargs):
    return SwinTransformer(model_name=model_name, input_shape=input_shape, window_size=window_size, embed_dim=embed_dim,
                           depths=depths, num_heads=num_heads, weights=weights, classes=classes, **kwargs)

