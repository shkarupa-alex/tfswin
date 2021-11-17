import numpy as np
import tensorflow as tf
from keras import backend, layers, models
from keras.applications import imagenet_utils
from keras.utils import conv_utils, data_utils, layer_utils
from .ape import AbsolutePositionEmbedding
from .basic import BasicLayer
from .embed import PatchEmbedding
from .norm import LayerNorm

BASE_WEIGHTS_PATH = ''
WEIGHTS_HASHES = {}


def SwinTransformer(
        model_name, input_shape, window_size, embed_dim, depths, num_heads,
        patch_size=4, patch_norm=True, use_ape=False, drop_rate=0., mlp_ratio=4., qkv_bias=True, qk_scale=None,
        attn_drop_rate=0., drop_path_rate=0.1,
        include_top=True, classes=1000, classifier_activation='softmax', pooling=None, weights='imagenet',
        input_tensor=None):
    """Instantiates the ResNet, ResNetV2, and ResNeXt architecture.

    TODO

    Returns:
      A `keras.Model` instance.
    """
    if not (weights in {'imagenet', None} or tf.io.gfile.exists(weights)):
        raise ValueError('The `weights` argument should be either `None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes not in {1000, 21843}:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top` as true, '
                         '`classes` should be 1000 or 21843 depending on model type')

    patch_size = conv_utils.normalize_tuple(patch_size, 2, 'patch_size')

    if input_tensor:
        try:
            backend.is_keras_tensor(input_tensor)
        except ValueError:
            raise ValueError(f'Expecting `input_tensor` to be a symbolic tensor instance. '
                             f'Got {input_tensor} of type {type(input_tensor)}')

    if input_tensor:
        tensor_shape = backend.int_shape(input_tensor)[1:]
        if input_shape and tensor_shape != input_shape:
            raise ValueError('Shape of `input_tensor` should equals to `input_shape` if both provided.')
        else:
            input_shape = tensor_shape

    # Determine proper input shape
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=224,
        min_size=32,
        data_format='channel_last',
        require_flatten=include_top,
        weights=weights)

    if input_tensor:
        if backend.is_keras_tensor(input_tensor):
            image = input_tensor
        else:
            image = layers.Input(tensor=input_tensor, shape=input_shape)
    else:
        image = layers.Input(shape=input_shape)

    # Define model pipeline
    x = PatchEmbedding(patch_size=patch_size[0], embed_dim=embed_dim, normalize=patch_norm, name='patch_embed')(image)

    if use_ape:
        x = AbsolutePositionEmbedding()(x)

    x = layers.Dropout(drop_rate, name='pos_drop')(x)

    patches_resolution = [input_shape[0] // patch_size[0], input_shape[1] // patch_size[1]]
    dpr = [x for x in np.linspace(0., drop_path_rate, sum(depths))]

    for i in range(len(depths)):
        resolution = (patches_resolution[0] // (2 ** i), patches_resolution[1] // (2 ** i))
        not_last = i != len(depths) - 1
        x = BasicLayer(
            dim=int(embed_dim * 2 ** i),
            input_resolution=resolution,
            depth=depths[i],
            num_heads=num_heads[i],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            path_drop=dpr[sum(depths[:i]):sum(depths[:i + 1])],
            downsample=not_last,
            name=f'layers.{i}')(x)

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
    if (weights == 'imagenet') and (model_name in WEIGHTS_HASHES):
        file_name = model_name + '_weights_tf_dim_ordering_tf_kernels.h5'
        file_hash = WEIGHTS_HASHES[model_name][0]
        weights_path = data_utils.get_file(
            file_name,
            BASE_WEIGHTS_PATH + file_name,
            cache_subdir='models',
            file_hash=file_hash)
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    if include_top:
        return model

    outputs = model.get_layer(name='avg_pool').output
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)

    return model


def SwinTransformerTiny224(**kwargs):
    return SwinTransformer(
        model_name='swin_tiny',
        input_shape=(224, 224, 3),
        window_size=7,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        **kwargs
    )


def SwinTransformerSmall224(**kwargs):
    return SwinTransformer(
        model_name='swin_small',
        input_shape=(224, 224, 3),
        window_size=7,
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        **kwargs
    )


def SwinTransformerBase224(**kwargs):
    # input_size = (224, 224, 3), window_size = 7, embed_dim = 128, depths = [2, 2, 18, 2], num_heads = [4, 8, 16, 32]
    return SwinTransformer(
        model_name='swin_base_224',
        input_shape=(224, 224, 3),
        window_size=7,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        **kwargs
    )


def SwinTransformerBase384(**kwargs):
    return SwinTransformer(
        model_name='swin_base_384',
        input_shape=(384, 384, 3),
        window_size=12,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        **kwargs
    )


def SwinTransformerLarge224(**kwargs):
    return SwinTransformer(
        model_name='swin_large_224',
        input_shape=(224, 224, 3),
        window_size=7,
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        **kwargs
    )


def SwinTransformerLarge384(**kwargs):
    return SwinTransformer(
        model_name='swin_large_384',
        input_shape=(384, 384, 3),
        window_size=12,
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        **kwargs
    )
