import torch
import torch.nn as nn
import numpy as np


def estimate_rank(weights, energy_threshold):
    rank = None
    _, s, _ = np.linalg.svd(weights)
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


def svd_decomposition(layer, energy_threshold):
    weights = layer.weight.cpu().data.numpy()
    is_bias = torch.is_tensor(layer.bias)
    rank = estimate_rank(weights, energy_threshold)

    U, S, Vt = np.linalg.svd(weights, full_matrices=True)
    w1 = np.dot(U[:, 0:rank], np.diag(np.sqrt(S[0:rank])))
    w0 = np.dot(np.diag(np.sqrt(S[0:rank])), Vt[0:rank, :])
    first_weights = torch.from_numpy(w0)
    second_weights = torch.from_numpy(w1)

    first_layer = nn.Linear(w0.shape[1], w0.shape[0], bias=False)
    second_layer = nn.Linear(w1.shape[1], w1.shape[0], bias=is_bias)

    first_layer.weight.data = first_weights
    second_layer.weight.data = second_weights

    if is_bias:
        second_layer.bias.data = layer.bias.data

    return nn.Sequential(first_layer, second_layer)
