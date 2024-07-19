import numpy as np
import tensorflow as tf
from keras.src import backend, layers, models
from keras.src.applications import imagenet_utils
from keras.src.dtype_policies import dtype_policy
from keras.src.ops import operation_utils
from keras.src.utils import get_file
from tfswin.ape import AbsoluteEmbedding
from tfswin.basic import BasicLayer
from tfswin.embed import PatchEmbedding
from tfswin.merge import PatchMerging

BASE_URL = 'https://github.com/shkarupa-alex/tfswin/releases/download/{}/swin{}_{}.h5'
WEIGHT_URLS = {
    'swin_tiny_224': BASE_URL.format('3.0.0', '', 'tiny_patch4_window7_224_22k'),
    'swin_small_224': BASE_URL.format('3.0.0', '', 'small_patch4_window7_224_22k'),
    'swin_base_224': BASE_URL.format('3.0.0', '', 'base_patch4_window7_224_22k'),
    'swin_base_384': BASE_URL.format('3.0.0', '', 'base_patch4_window12_384_22k'),
    'swin_large_224': BASE_URL.format('3.0.0', '', 'large_patch4_window7_224_22k'),
    'swin_large_384': BASE_URL.format('3.0.0', '', 'large_patch4_window12_384_22k'),

    'swin2_tiny_256': BASE_URL.format('3.0.0', 'v2', 'tiny_patch4_window16_256'),
    'swin2_small_256': BASE_URL.format('3.0.0', 'v2', 'small_patch4_window16_256'),
    'swin2_base_256': BASE_URL.format('3.0.0', 'v2', 'base_patch4_window12to16_192to256_22kto1k_ft'),
    'swin2_base_384': BASE_URL.format('3.0.0', 'v2', 'base_patch4_window12to24_192to384_22kto1k_ft'),
    'swin2_large_256': BASE_URL.format('3.0.0', 'v2', 'large_patch4_window12to16_192to256_22kto1k_ft'),
    'swin2_large_384': BASE_URL.format('3.0.0', 'v2', 'large_patch4_window12to24_192to384_22kto1k_ft')
}
WEIGHT_HASHES = {
    'swin_tiny_224': '06588c1314ec16f5d46b6381431c1fc9355f82c4eda0f34e7e57ee32048b6d9a',
    'swin_small_224': '49d799e6b860d97377e2d67f866d691f6ee8e02dd6e6d9c485eb3ee0a0076508',
    'swin_base_224': '61792331fce60c47cf7d2bd8e77c8496245cced095a698ab84b275fc20aab32c',
    'swin_base_384': '928ff42574b6c24e55be56563a73419d51e38a0488d72c075c040de8dafd6df2',
    'swin_large_224': '22bf2dc2e5bd99a490b7cd2362519fbe672a576c78c275d12ad17461e75664fd',
    'swin_large_384': '9deaef34b99aa58240cb5f9fa383838e39646e7d23c77942cb3ab320e175124c',

    'swin2_tiny_256': '6416e13f50c46dcd2bf9252d548053fe6443b2a6b4d4bc69cab295923092e3cd',
    'swin2_small_256': '669f81de05bb5fb5419924f1c2847e5ed9ebf2e353ee06699101cc5e124dec4c',
    'swin2_base_256': '50cc998725e9024c9d4da6951f41a6205b4aec104d17f277a77ebf7020074af3',
    'swin2_base_384': '1ba46e24461aab919cb7997cee17de1259d13917d84366a836551d9fe920c6de',
    'swin2_large_256': '87b4cfc35ba6c3fc9de52184c37dfb2ec280e22052730678b1f9c6c4e5719448',
    'swin2_large_384': '9d83698a8cc05eb1ff978665ea8891d5471f8ea40a1b079bf93d241e31306abc'
}


def SwinTransformer(
        pretrain_size, window_size, embed_dim, depths, num_heads, patch_size=4, patch_norm=True, use_ape=False,
        drop_rate=0., mlp_ratio=4., qkv_bias=True, qk_scale=None, attn_drop=0., path_drop=0.1,
        window_pretrain=None, swin_v2=False, model_name='swin', include_top=True, weights=None,
        input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation='softmax',
        include_preprocessing=True):
    """Instantiates the Swin Transformer architecture.

    Args:
      pretrain_size: height/width of input image for pretraining.
      window_size: window partition size.
      embed_dim: patch embedding dimension.
      depths: depth of each Swin Transformer layer.
      num_heads: number of attention heads.
      patch_size: patch size used to divide input image.
      patch_norm: whether to add normalization after patch embedding.
      use_ape: whether to add absolute position embedding to the patch embedding.
      drop_rate: dropout rate.
      mlp_ratio: ratio of mlp hidden units to embedding units.
      qkv_bias: whether to add a learnable bias to query, key, value.
      qk_scale: override default qk scale of head_dim ** -0.5 if set.
      attn_drop: attention dropout rate.
      path_drop: stochastic depth rate.
      window_pretrain: window partition size used in pretraining.
      swin_v2: whether to use V2 model architecture.
      model_name: model name.
      include_top: whether to include the fully-connected layer at the top of the network.
      weights: one of `None` (random initialization), 'imagenet' (pre-training on ImageNet or ImageNet 21k), or the
        path to the weights file to be loaded.
      input_tensor: tensor (i.e. output of `layers.Input()`) to use as image input for the model.
      input_shape: shape tuple without batch dimension. Used to create input layer if `input_tensor` not provided.
      pooling: optional pooling mode for feature extraction when `include_top` is `False`.
        - `None` means that the output of the model will be the 3D tensor output of the last layer.
        - `avg` means that global average pooling will be applied to the output of the last layer, and thus the output
          of the model will be a 2D tensor.
        - `max` means that global max pooling will be applied.
      classes: optional number of classes to classify images into, only to be specified if `include_top` is True.
      classifier_activation: the activation function to use on the "top" layer. Ignored unless `include_top=True`.
        When loading pretrained weights, `classifier_activation` can only be `None` or `"softmax"`.
      include_preprocessing: Boolean, whether to include the preprocessing layer at the bottom of the network.
        Note: Input image is normalized by ImageNet mean and standard deviation. Defaults to `True`.

    Returns:
      A `keras.src.Model` instance.
    """
    if not (weights in {'imagenet', None} or tf.io.gfile.exists(weights)):
        raise ValueError('The `weights` argument should be either `None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes not in {1000, 21841}:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top` as true, '
                         '`classes` should be 1000 or 21841 depending on model type')

    if input_tensor is not None:
        try:
            backend.is_keras_tensor(input_tensor)
        except ValueError:
            raise ValueError(f'Expecting `input_tensor` to be a symbolic tensor instance. '
                             f'Got {input_tensor} of type {type(input_tensor)}')

    if input_tensor is not None:
        tensor_shape = backend.int_shape(input_tensor)[1:]
        if input_shape and tensor_shape != input_shape:
            raise ValueError('Shape of `input_tensor` should equals to `input_shape` if both provided.')
        else:
            input_shape = tensor_shape

    # Determine proper input shape
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=pretrain_size,
        min_size=32,
        data_format='channel_last',
        require_flatten=False,
        weights=weights)
    input_dtype = dtype_policy.dtype_policy().compute_dtype

    if input_tensor is not None:
        if backend.is_keras_tensor(input_tensor):
            image = input_tensor
        else:
            image = layers.Input(tensor=input_tensor, shape=input_shape, dtype=input_dtype)
    else:
        image = layers.Input(shape=input_shape, dtype=input_dtype)

    x = image

    if include_preprocessing:
        imagenet_mean = np.array([0.485, 0.456, 0.406], 'float32') * 255.
        imagenet_var = (np.array([0.229, 0.224, 0.225], 'float32') * 255.) ** 2
        x = layers.Normalization(mean=imagenet_mean.tolist(), variance=imagenet_var.tolist(), name='normalize')(x)

    # Define model pipeline
    x = PatchEmbedding(patch_size=patch_size, embed_dim=embed_dim, normalize=patch_norm, name='patch_embed')(x)

    if use_ape:
        pretrain_size_ = operation_utils.compute_conv_output_shape(
            (None, pretrain_size, pretrain_size, 3), embed_dim, (patch_size, patch_size), padding='same',
            strides=patch_size, dilation_rate=1)
        x = AbsoluteEmbedding(pretrain_size_[1])(x)

    x = layers.Dropout(drop_rate, name='pos_drop')(x)

    path_drops = np.linspace(0., path_drop, sum(depths))

    if not swin_v2:
        window_pretrain = np.minimum(window_size, pretrain_size // 2 ** np.arange(2, 6)).tolist()
    elif window_pretrain is None and swin_v2:
        window_pretrain = [0] * len(depths)

    for i in range(len(depths)):
        path_drop = path_drops[sum(depths[:i]):sum(depths[:i + 1])].tolist()
        not_last = i != len(depths) - 1

        x = BasicLayer(depth=depths[i], num_heads=num_heads[i], window_size=window_size, mlp_ratio=mlp_ratio,
                       qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop, path_drop=path_drop,
                       window_pretrain=window_pretrain[i], swin_v2=swin_v2, name=f'layers.{i}')(x)
        if not_last:
            x = PatchMerging(swin_v2=swin_v2, name=f'layers.{i}.downsample')(x)

    x = layers.LayerNormalization(epsilon=1.001e-5, name='norm')(x)

    if include_top or pooling in {None, 'avg'}:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling2D(name='max_pool')(x)
    else:
        raise ValueError(f'Expecting pooling to be one of None/avg/max. Found: {pooling}')

    imagenet_utils.validate_activation(classifier_activation, weights)
    x = layers.Dense(classes, name='head')(x)
    x = layers.Activation(classifier_activation, dtype='float32', name='pred')(x)

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = operation_utils.get_source_inputs(input_tensor)
    else:
        inputs = image

    # Create model.
    model = models.Functional(inputs=inputs, outputs=x, name=model_name)

    # Load weights.
    if 'imagenet' == weights and model_name in WEIGHT_URLS:
        weights_url = WEIGHT_URLS[model_name]
        weights_hash = WEIGHT_HASHES[model_name]
        weights_path = get_file(origin=weights_url, file_hash=weights_hash, cache_subdir='tfswin')
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
    model = models.Functional(inputs=inputs, outputs=outputs, name=model_name)

    return model


def SwinTransformerTiny224(model_name='swin_tiny_224', pretrain_size=224, window_size=7, embed_dim=96,
                           depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), path_drop=0.2, weights='imagenet',
                           classes=21841, **kwargs):
    return SwinTransformer(model_name=model_name, pretrain_size=pretrain_size, window_size=window_size,
                           embed_dim=embed_dim, depths=depths, num_heads=num_heads, path_drop=path_drop,
                           weights=weights, classes=classes, **kwargs)


def SwinTransformerSmall224(model_name='swin_small_224', pretrain_size=224, window_size=7, embed_dim=96,
                            depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), path_drop=0.3, weights='imagenet',
                            classes=21841, **kwargs):
    return SwinTransformer(model_name=model_name, pretrain_size=pretrain_size, window_size=window_size,
                           embed_dim=embed_dim, depths=depths, num_heads=num_heads, path_drop=path_drop,
                           weights=weights, classes=classes, **kwargs)


def SwinTransformerBase224(model_name='swin_base_224', pretrain_size=224, window_size=7, embed_dim=128,
                           depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), path_drop=0.5, weights='imagenet',
                           classes=21841, **kwargs):
    return SwinTransformer(model_name=model_name, pretrain_size=pretrain_size, window_size=window_size,
                           embed_dim=embed_dim, depths=depths, num_heads=num_heads, path_drop=path_drop,
                           weights=weights, classes=classes, **kwargs)


def SwinTransformerBase384(model_name='swin_base_384', pretrain_size=384, window_size=12, embed_dim=128,
                           depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), weights='imagenet', classes=21841, **kwargs):
    return SwinTransformer(model_name=model_name, pretrain_size=pretrain_size, window_size=window_size,
                           embed_dim=embed_dim, depths=depths, num_heads=num_heads, weights=weights, classes=classes,
                           **kwargs)


def SwinTransformerLarge224(model_name='swin_large_224', pretrain_size=224, window_size=7, embed_dim=192,
                            depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48), weights='imagenet', classes=21841,
                            **kwargs):
    return SwinTransformer(model_name=model_name, pretrain_size=pretrain_size, window_size=window_size,
                           embed_dim=embed_dim, depths=depths, num_heads=num_heads, weights=weights, classes=classes,
                           **kwargs)


def SwinTransformerLarge384(model_name='swin_large_384', pretrain_size=384, window_size=12, embed_dim=192,
                            depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48), weights='imagenet', classes=21841,
                            **kwargs):
    return SwinTransformer(model_name=model_name, pretrain_size=pretrain_size, window_size=window_size,
                           embed_dim=embed_dim, depths=depths, num_heads=num_heads, weights=weights, classes=classes,
                           **kwargs)


def SwinTransformerV2Tiny256(model_name='swin2_tiny_256', pretrain_size=256, window_size=16, embed_dim=96,
                             depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), path_drop=0.2, weights='imagenet',
                             **kwargs):
    return SwinTransformer(model_name=model_name, pretrain_size=pretrain_size, window_size=window_size,
                           embed_dim=embed_dim, depths=depths, num_heads=num_heads, path_drop=path_drop, swin_v2=True,
                           weights=weights, **kwargs)


def SwinTransformerV2Small256(model_name='swin2_small_256', pretrain_size=256, window_size=16, embed_dim=96,
                              depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), path_drop=0.3, weights='imagenet',
                              **kwargs):
    return SwinTransformer(model_name=model_name, pretrain_size=pretrain_size, window_size=window_size,
                           embed_dim=embed_dim, depths=depths, num_heads=num_heads, path_drop=path_drop, swin_v2=True,
                           weights=weights, **kwargs)


def SwinTransformerV2Base256(model_name='swin2_base_256', pretrain_size=256, window_size=16, embed_dim=128,
                             depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), path_drop=0.2,
                             window_pretrain=(12, 12, 12, 6), weights='imagenet', **kwargs):
    return SwinTransformer(model_name=model_name, pretrain_size=pretrain_size, window_size=window_size,
                           embed_dim=embed_dim, depths=depths, num_heads=num_heads, path_drop=path_drop,
                           window_pretrain=window_pretrain, swin_v2=True, weights=weights, **kwargs)


def SwinTransformerV2Base384(model_name='swin2_base_384', pretrain_size=384, window_size=24, embed_dim=128,
                             depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), path_drop=0.2,
                             window_pretrain=(12, 12, 12, 6), weights='imagenet', **kwargs):
    return SwinTransformer(model_name=model_name, pretrain_size=pretrain_size, window_size=window_size,
                           embed_dim=embed_dim, depths=depths, num_heads=num_heads, path_drop=path_drop,
                           window_pretrain=window_pretrain, swin_v2=True, weights=weights, **kwargs)


def SwinTransformerV2Large256(model_name='swin2_large_256', pretrain_size=256, window_size=16, embed_dim=192,
                              depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48), path_drop=0.2,
                              window_pretrain=(12, 12, 12, 6), weights='imagenet', **kwargs):
    return SwinTransformer(model_name=model_name, pretrain_size=pretrain_size, window_size=window_size,
                           embed_dim=embed_dim, depths=depths, num_heads=num_heads, path_drop=path_drop,
                           window_pretrain=window_pretrain, swin_v2=True, weights=weights, **kwargs)


def SwinTransformerV2Large384(model_name='swin2_large_384', pretrain_size=384, window_size=24, embed_dim=192,
                              depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48), path_drop=0.2,
                              window_pretrain=(12, 12, 12, 6), weights='imagenet', **kwargs):
    return SwinTransformer(model_name=model_name, pretrain_size=pretrain_size, window_size=window_size,
                           embed_dim=embed_dim, depths=depths, num_heads=num_heads, path_drop=path_drop,
                           window_pretrain=window_pretrain, swin_v2=True, weights=weights, **kwargs)
