import matlab.engine
import numpy as np
import torch
import torch.nn as nn


def estimate_rank(mat_weights, energy_threshold, eng):
    rank = None
    unfold_mat = eng.unfolding(mat_weights, 1)
    unfold = np.asarray(unfold_mat, dtype=np.float32)

    _, s, _ = np.linalg.svd(unfold)
    total_sum = np.sum(s**2)
    s_sum = 0
    count = 0

    for i in s:
        s_sum += i**2
        count += 1
        energy = s_sum / total_sum
        if energy > energy_threshold:
            rank = count
            break

    return rank


def hosvd1(layer, eng, energy_threshold):
    is_bias = torch.is_tensor(layer.bias)
    weights = layer.weight.cpu().data.numpy()
    toMatlabWeights = weights.tolist()
    mat_weights = matlab.double(toMatlabWeights)
    R = estimate_rank(mat_weights, energy_threshold, eng)
    U, T = eng.hosvd1(mat_weights, R, nargout=2)

    first = np.asarray(T, dtype=np.float32)
    second = np.asarray(U, dtype=np.float32)

    inChannels = layer.in_channels
    outChannels = layer.out_channels

    firstWeights = first if first.ndim > 2 else np.expand_dims(first, axis=(2, 3))
    secondWeights = np.expand_dims(second, axis=(2, 3))

    first_layer = nn.Conv2d(
        in_channels=inChannels,
        out_channels=R,
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        bias=False,
    )

    second_layer = nn.Conv2d(
        in_channels=R,
        out_channels=outChannels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=is_bias,
    )

    first_layer.weight.data = torch.from_numpy(firstWeights)
    if is_bias:
        second_layer.weight.data = torch.from_numpy(secondWeights)

    return nn.Sequential(first_layer, second_layer)
