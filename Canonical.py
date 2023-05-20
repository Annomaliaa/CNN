import tensorly as tl
from tensorly.decomposition import parafac
import numpy as np
import torch
import torch.nn as nn
from VBMF import VBMF

def estimate_rank_cp(layer):
    """Estimate the rank of a tensor factorized using CP decomposition."""

    # Get the weight tensor of the layer
    weights = layer.weight.cpu().data.numpy()

    print("Weight Tensor", weights.shape)

    # Convert the weight tensor into a three-dimensional tensor
    T, S, D1, D12 = weights.shape

    D2 = D1 * D12

    weights_tensor = tl.tensor(weights, dtype=tl.float32).reshape((T, S, D2))



    print("Dimension T", T)
    print("Dimension S", S)
    print("Dimension D", D2)

    print("Threedimensional tensor", weights_tensor.shape)

    # Matricize the tensor along each of its three dimensions
    matricized_0 = tl.unfold(weights_tensor, mode=0)
    matricized_0_CP = matricized_0.reshape((T, S*D2))

    print(matricized_0_CP.shape)

    matricized_1 = tl.unfold(weights_tensor, mode=1)
    matricized_1_CP = matricized_1.reshape((S, T*D2))
    matricized_2 = tl.unfold(weights_tensor, mode=2)
    matricized_2_CP = matricized_2.reshape((D2, T*S))

    # Estimate the rank of each matrix using VBMF
    _, diag_0, _, _ = VBMF.EVBMF(matricized_0_CP)
    _, diag_1, _, _ = VBMF.EVBMF(matricized_1_CP)
    _, diag_2, _, _ = VBMF.EVBMF(matricized_2_CP)

    # Select the maximum rank among the three ranks
    ranks = [diag_0.shape[0], diag_1.shape[0], diag_2.shape[0]]

    print("All ranks", ranks)

    max_rank = max(ranks)

    print("Maximum ranks for CP Decompositiom", max_rank)

    return max_rank


def cp_decomposition_conv_layer(layer):
    """ Gets a conv layer and a target rank,
        returns a nn.Sequential object with the decomposition """

    rank = estimate_rank_cp(layer)
    # Perform CP decomposition on the layer weight tensorly.

    weights = layer.weight.cpu().data.numpy()

    #last, first, vertical, horizontal = parafac(weights, rank=rank, init='svd')

    CP_tensor, errors = parafac(weights, rank=rank, init='svd', tol=10e-3, verbose=2, return_errors=True)

    print("Reconstructin error", errors)

    [weight, factors] = CP_tensor

    [last, first, vertical, horizontal] = factors

    for i in range(len(factors)):
        print(factors[i].shape)

    print(last.shape)
    print(first.shape)
    print(vertical.shape)
    print(horizontal.shape)

    pointwise_s_to_r_layer = torch.nn.Conv2d(in_channels=first.shape[0], out_channels=first.shape[1], kernel_size=1, stride=1, padding=0,
                                             dilation=layer.dilation, bias=False)

    depthwise_vertical_layer = torch.nn.Conv2d(in_channels=vertical.shape[1],
                                               out_channels=vertical.shape[1], kernel_size=(vertical.shape[0], 1),
                                               stride=1, padding=(layer.padding[0], 0), dilation=layer.dilation,
                                               groups=vertical.shape[1], bias=False)

    depthwise_horizontal_layer = \
        torch.nn.Conv2d(in_channels=horizontal.shape[1], out_channels=horizontal.shape[1],
                        kernel_size=(1, horizontal.shape[0]), stride=layer.stride,
                        padding=(0, layer.padding[0]),
                        dilation=layer.dilation, groups=horizontal.shape[1], bias=False)

    pointwise_r_to_t_layer = torch.nn.Conv2d(in_channels=last.shape[1], out_channels=last.shape[0], kernel_size=1, stride=1,
                                             padding=0, dilation=layer.dilation, bias=True)

    if layer.bias is not None:
        pointwise_r_to_t_layer.bias.data = layer.bias.data

    last = torch.from_numpy(last)
    first = torch.from_numpy(first)
    vertical = torch.from_numpy(vertical)
    horizontal = torch.from_numpy(horizontal)

    depthwise_horizontal_layer.weight.data = torch.transpose(horizontal, 1, 0).unsqueeze(1).unsqueeze(1)
    depthwise_vertical_layer.weight.data = torch.transpose(vertical, 1, 0).unsqueeze(1).unsqueeze(-1)
    pointwise_s_to_r_layer.weight.data = torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    pointwise_r_to_t_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)

    new_layers = [pointwise_s_to_r_layer, depthwise_vertical_layer, depthwise_horizontal_layer, pointwise_r_to_t_layer]

    return nn.Sequential(*new_layers)
