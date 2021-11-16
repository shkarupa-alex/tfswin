import torch
import tensorflow as tf


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


def main(m):
    weights_file = '/Users/alex/Develop/tfswin/swin_tiny_patch4_window7_224.pth'
    state_dict = torch.load(weights_file, map_location=torch.device("cpu"))

    res = []
    for w in m.weights:
        name = convert_name(w.name)
        if 'attn.relative_position_index' in name or 'attn_mask' in name:
            res.append(w.numpy())
        else:
            assert name in state_dict['model']
            weight = state_dict['model'][name].numpy()
            weight = convert_weight(weight, name)
            res.append(weight)

    m.set_weights(res)

