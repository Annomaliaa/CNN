import tensorly as tl
from matplotlib import pyplot as plt
from tensorly.decomposition import parafac, partial_tucker
import numpy as np
import torch
import torch.nn as nn
from VBMF import VBMF



def estimate_ranks(layer):
    """ Unfold the 2 modes of the Tensor the decomposition will
    be performed on, and estimates the ranks of the matrices using VBMF
    """

    weights = layer.weight.cpu().data.numpy()
    unfold_0 = tl.unfold(weights, 0)
    unfold_0 = np.asarray(unfold_0, dtype=np.float32)

    unfold_1 = tl.unfold(weights, 1)
    unfold_1 = np.asarray(unfold_1, dtype=np.float32)

    _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
    _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
    ranks = [diag_0.shape[0], diag_1.shape[1]]
    return ranks

def get_original_factors_conv_layer(layer):
    """
    Given a convolutional layer, returns its factors before any decomposition.
    """
    weight = layer.weight.data
    out_channels, in_channels, kernel_size1, kernel_size2 = weight.size()

    # Compute SVD of the weight tensor
    u, s, v = torch.svd(weight.view(out_channels, -1))

    # Compute the size of the factor matrices
    k = min(out_channels, in_channels, kernel_size1 * kernel_size2)

    # Truncate the factor matrices
    u = u[:, :k]
    s = s[:k]
    v = v[:, :k]

    print(weight.shape)

    # Reshape the factor matrices to match the weight tensor
    u = u.view(out_channels, -1, k).permute(2, 0, 1)
    s = torch.diag(s)
    v = v.view(in_channels, kernel_size1, kernel_size2, k).permute(3, 0, 1, 2)

    # Return the factor matrices as a tuple
    return (u, s, v)

def tucker_decomposition_conv_layer(layer):

    ranks = estimate_ranks(layer)
    weights = layer.weight.cpu().data.numpy()

    [R1, R2] = ranks
    factors_original = get_original_factors_conv_layer(layer)

    print("Original Factors", factors_original[0].shape)
    print("Original Factors", factors_original[1].shape)
    print("Original Factors", factors_original[2].shape)

    print(layer, "VBMF Estimated ranks:", ranks)
    print(layer, "Weights:", weights.shape)

    weight = layer.weight.cpu().data.numpy()

    (core, factors), rec_errors = partial_tucker(weight, modes = [0,1], rank=ranks, init='svd', tol=10e-4, svd='randomized_svd',  verbose=2)

    print(rec_errors)

    #[first, last] = factors

    [last, first] = factors
    print("First Tensor", first.shape)
    print("Last Tensor", last.shape)
    print("Core Tensor", core.shape)

    first_layer = nn.Conv2d(in_channels=first.shape[0],
                            out_channels=first.shape[1],
                            kernel_size=1,
                            padding=0,
                            bias=False)

    core_layer = nn.Conv2d(in_channels=core.shape[1],
                           out_channels=core.shape[0],
                           kernel_size=layer.kernel_size,
                           stride=layer.stride,
                           padding=layer.padding,
                           dilation=layer.dilation,
                           bias=False)

    last_layer = nn.Conv2d(in_channels=last.shape[1],
                           out_channels=last.shape[0],
                           kernel_size=1,
                           padding=0,
                           bias=True)

    if layer.bias is not None:
        last_layer.bias.data = layer.bias.data

    # Visualize the factors using matplotlib
    fig, axs = plt.subplots(nrows=len(factors))
    for i, factor in enumerate(factors):
        axs[i].imshow(factor)
        axs[i].set_title(f'Factor {i + 1}')
    fig.tight_layout()
    plt.show()

    #Reconstructed weights
    compressed_weights = tl.tucker_to_tensor([core, factors], )

    # Compute the reconstruction error
    error = np.linalg.norm(weight - compressed_weights) / np.linalg.norm(weight)
    print("Reconstruction error:", error)

    print("Reconstructed weights", compressed_weights.shape)


    print("First Tensor", first.shape)
    print("Last Tensor", last.shape)
    print("Core Tensor", core.shape)

    print(first_layer.weight.shape)
    print(last_layer.weight.shape)
    print(core_layer.weight.shape)

    """Tried compressed weights to fit as the new but was the wrong approach"""
    #first_layer.weight.data = torch.transpose(compressed_weights[0], 0, 1)
    #last_layer.weight.data = torch.transpose(compressed_weights[2], 0, 1, 3, 2)
    #last_layer.bias.data = compressed_weights[2][:, 0, 0, :]
    #core_layer.weight.data = compressed_weights[1]

    """This are the good ones work with VGG16"""
    #first = np.asarray(first, np.float32)
    #first = torch.from_numpy(first)
    #last = np.asarray(last, np.float32)
    #last = torch.from_numpy(last)
    #core = np.asarray(core, np.float32)
    #core = torch.from_numpy(core)

    #first_layer.weight.data = torch.transpose(last, 0, 1).unsqueeze(-1).unsqueeze(-1)
    #last_layer.weight.data = first.unsqueeze(-1).unsqueeze(-1)
    #core_layer.weight.data = core

    """THIS ARE THE NEW ONES - works"""

    first = torch.from_numpy(first)
    last = torch.from_numpy(last)
    core = torch.from_numpy(core)

    # Continue with the rest of your code
    fk = first.t_().unsqueeze_(-1).unsqueeze_(-1)
    lk = last.unsqueeze_(-1).unsqueeze_(-1)

    first_layer.weight.data = fk
    last_layer.weight.data = lk
    core_layer.weight.data = core

    print("First Layer Weight Data", first_layer.weight.data.shape)
    print("Last Layer Weight Data", last_layer.weight.data.shape)
    print("Core Layer Weight Data", core_layer.weight.data.shape)

    print("Weights of the original model", weight.shape)

    new_layers = [first_layer, core_layer, last_layer]
    #return nn.Sequential(*new_layers) for vgg16
    # for resnet20
    return nn.Sequential(*new_layers)

