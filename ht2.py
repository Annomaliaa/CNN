import matlab
import numpy as np
import torch
from torch import nn
from thop import profile


def estimate_ranks(mat_weights, energy_threshold, eng):
    ranks = []

    unfold_0_mat = eng.unfolding(mat_weights, 1)
    unfold_0 = np.asarray(unfold_0_mat, dtype=np.float32)

    unfold_1_mat = eng.unfolding(mat_weights, 3)
    unfold_1 = np.asarray(unfold_1_mat, dtype=np.float32)

    unfold_2_mat = eng.can_matricization(mat_weights, 1, 2)
    unfold_2 = np.asarray(unfold_2_mat, dtype=np.float32)

    for unfold in (unfold_0, unfold_1, unfold_2):
        _, s, _ = np.linalg.svd(unfold)
        total_sum = np.sum(s ** 2)
        s_sum = 0
        count = 0

        for i in s:
            s_sum += i ** 2
            count += 1
            energy = s_sum / total_sum
            if energy > energy_threshold:
                ranks.append(count)
                break

    return ranks


def ht2(layer, eng, energy_threshold):
    is_bias = torch.is_tensor(layer.bias)
    weights = layer.weight.cpu().data.numpy()
    weights = np.moveaxis(weights, 0, 2)

    mat_weights = matlab.double(weights.tolist())
    R1, R3, R13 = estimate_ranks(mat_weights, energy_threshold, eng)

    factors = eng.ht2_conv_decomposition(mat_weights, R1, R3, R13)

    first = np.asarray(factors[0], dtype=np.float32)
    fourth = np.asarray(factors[1], dtype=np.float32)
    second = np.asarray(factors[2], dtype=np.float32)
    third = np.asarray(factors[3], dtype=np.float32)

    print("First", first.shape)
    print("Fourth", fourth.shape)
    print("Second", second.shape)
    print("Third", third.shape)

    first_weights = np.expand_dims(np.swapaxes(first, 0, 1), axis=(2, 3))
    second_weights = np.expand_dims(np.moveaxis(second, -1, 0), axis=3)
    third_weights = np.expand_dims(np.swapaxes(third, 1, 2), axis=2)
    fourth_weights = np.expand_dims(fourth, axis=(2, 3))

    print("First Weights", first_weights.shape)
    print("Fourth Weights", fourth_weights.shape)
    print("Second Weights", second_weights.shape)
    print("Third Weights", third_weights.shape)

    first_layer = nn.Conv2d(
        in_channels=layer.in_channels,
        out_channels=R1,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=False,
    )

    second_layer = nn.Conv2d(
        in_channels=R1,
        out_channels=R13,
        kernel_size=(layer.kernel_size[0], 1),
        stride=(layer.stride[0], 1),
        padding=(layer.padding[0], 0),
        dilation=layer.dilation,
        bias=False,
    )

    third_layer = nn.Conv2d(
        in_channels=R13,
        out_channels=R3,
        kernel_size=(1, layer.kernel_size[1]),
        stride=(1, layer.stride[1]),
        padding=(0, layer.padding[1]),
        dilation=layer.dilation,
        bias=False,
    )

    fourth_layer = nn.Conv2d(
        in_channels=R3,
        out_channels=layer.out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=is_bias,
    )

    print(first_layer.weight.shape)
    print(second_layer.weight.shape)
    print(third_layer.weight.shape)
    print(fourth_layer.weight.shape)

    first_layer.weight.data = torch.from_numpy(first_weights)
    second_layer.weight.data = torch.from_numpy(second_weights)
    third_layer.weight.data = torch.from_numpy(third_weights)
    fourth_layer.weight.data = torch.from_numpy(fourth_weights)

    print("First Weights", first_layer.weight.data.shape)
    print("Fourth Weights", fourth_layer.weight.data.shape)
    print("Second Weights", second_layer.weight.data.shape)
    print("Third Weights", third_layer.weight.data.shape)

    if is_bias:
        fourth_layer.bias.data = layer.bias.data

    return nn.Sequential(*[first_layer, second_layer, third_layer, fourth_layer])