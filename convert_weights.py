#!/usr/bin/env python3
import argparse
import os
import tfswin
import torch
from tf_keras.src.utils.data_utils import get_file

BASE_URL = 'https://github.com/SwinTransformer/storage/releases/download/v{}/{}.pth'
CHECKPOINTS = {
    'v1_tiny_224': BASE_URL.format('1.0.8', 'swin_tiny_patch4_window7_224_22k'),
    'v1_small_224': BASE_URL.format('1.0.8', 'swin_small_patch4_window7_224_22k'),
    'v1_base_224': BASE_URL.format('1.0.0', 'swin_base_patch4_window7_224_22k'),
    'v1_base_384': BASE_URL.format('1.0.0', 'swin_base_patch4_window12_384_22k'),
    'v1_large_224': BASE_URL.format('1.0.0', 'swin_large_patch4_window7_224_22k'),
    'v1_large_384': BASE_URL.format('1.0.0', 'swin_large_patch4_window12_384_22k'),

    'v2_tiny_256': BASE_URL.format('2.0.0', 'swinv2_tiny_patch4_window16_256'),
    'v2_small_256': BASE_URL.format('2.0.0', 'swinv2_small_patch4_window16_256'),
    'v2_base_256': BASE_URL.format('2.0.0', 'swinv2_base_patch4_window12to16_192to256_22kto1k_ft'),
    'v2_base_384': BASE_URL.format('2.0.0', 'swinv2_base_patch4_window12to24_192to384_22kto1k_ft'),
    'v2_large_256': BASE_URL.format('2.0.0', 'swinv2_large_patch4_window12to16_192to256_22kto1k_ft'),
    'v2_large_384': BASE_URL.format('2.0.0', 'swinv2_large_patch4_window12to24_192to384_22kto1k_ft')
}
TF_MODELS = {
    'v1_tiny_224': tfswin.SwinTransformerTiny224,
    'v1_small_224': tfswin.SwinTransformerSmall224,
    'v1_base_224': tfswin.SwinTransformerBase224,
    'v1_base_384': tfswin.SwinTransformerBase384,
    'v1_large_224': tfswin.SwinTransformerLarge224,
    'v1_large_384': tfswin.SwinTransformerLarge384,

    'v2_tiny_256': tfswin.SwinTransformerV2Tiny256,
    'v2_small_256': tfswin.SwinTransformerV2Small256,
    'v2_base_256': tfswin.SwinTransformerV2Base256,
    'v2_base_384': tfswin.SwinTransformerV2Base384,
    'v2_large_256': tfswin.SwinTransformerV2Large256,
    'v2_large_384': tfswin.SwinTransformerV2Large384
}


def convert_name(name):
    name = name.replace(':0', '').replace('/', '.')
    name = name.replace('kernel', 'weight')
    name = name.replace('gamma', 'weight').replace('beta', 'bias')

    return name


def convert_weight(weight, name):
    if '.weight' in name and 4 == len(weight.shape):
        weight = weight.transpose([2, 3, 1, 0])

    if '.weight' in name and 2 == len(weight.shape):
        weight = weight.T

    return weight


if '__main__' == __name__:
    parser = argparse.ArgumentParser(description='Swin Transformer weight conversion from PyTorch to TensorFlow')
    parser.add_argument(
        'model_type',
        type=str,
        choices=list(CHECKPOINTS.keys()),
        help='Model checkpoint to load')
    parser.add_argument(
        'out_path',
        type=str,
        help='Path to save TensorFlow model weights')

    argv, _ = parser.parse_known_args()
    assert os.path.exists(argv.out_path) and os.path.isdir(argv.out_path), 'Wrong output path'

    weights_path = get_file(
        fname=None,
        origin=CHECKPOINTS[argv.model_type],
        cache_subdir='',
        cache_dir=argv.out_path)
    weights_torch = torch.load(weights_path, map_location=torch.device('cpu'))

    model = TF_MODELS[argv.model_type](weights=None)

    weights_tf = []
    for w in model.weights:
        name = convert_name(w.name)
        assert name in weights_torch['model'], f'Can\'t find weight {name} in checkpoint'

        weight = weights_torch['model'].pop(name).numpy()
        weight = convert_weight(weight, name)

        weights_tf.append(weight)

    model.set_weights(weights_tf)
    model.save_weights(weights_path.replace('.pth', '.h5'), save_format='h5')
