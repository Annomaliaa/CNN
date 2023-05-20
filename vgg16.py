import matlab.engine
import torchvision
from torchview import draw_graph
import torch
from torch import nn
from torchvision import models
from ht2 import ht2
from hosvd1 import hosvd1
from svd_decomposition import svd_decomposition
import matlab.engine
from TuckerforResnet import tucker_decomposition_conv_layer
from timeit import default_timer as timer
from torch.profiler import profile, record_function, ProfilerActivity
from ptflops import get_model_complexity_info
from flopth import flopth
import sys
import tensorly as tl
import numpy as np
from svd_decomposition import svd_decomposition


sys.path.append("C:/Program Files/MATLAB/R2022b/bin/matlab.exe")
sys.path.append("C:/Users/Ania/Downloads/tensor_toolbox-v3.5")


if __name__ == "__main__":
    energy = 0.78 #Try diffrent thredshold level
    net = models.vgg16_bn(True)
    eng = matlab.engine.start_matlab()
    start = timer()

    eng.addpath("C:/Program Files/MATLAB/R2022b/bin/matlab")
    eng.addpath("C:/Users/Ania/Documents/MATLAB/matlab")

    import matlab.engine
    import sys


    sys.path.append("C:/Program Files/MATLAB/R2022b/bin/matlab.exe")
    sys.path.append("C:/Users/Ania/Downloads/tensor_toolbox-v3.5")

    eng = matlab.engine.start_matlab()

    eng.addpath("C:/Program Files/MATLAB/R2022b/bin/matlab")
    eng.addpath("C:/Users/Ania/Documents/MATLAB/matlab")

    energy = 0.8

    for key, layer in net.features._modules.items():
        if key == "0":
            decomposed = hosvd1(layer, eng, energy)
            net.features._modules[key] = decomposed

        if isinstance(layer, nn.modules.conv.Conv2d) and key != "0":
            #decomposed_tucker = tucker_decomposition_conv_layer(layer)
            decomposed_tucker = tucker_decomposition_conv_layer(layer)
            net.features._modules[key] = decomposed_tucker
            print("Decomposed Layer", decomposed_tucker)
            original_weights = layer.weight.cpu().data.numpy()
            decomposed_weights = decomposed_tucker[1].weight.data.cpu().numpy()


    for key, layer in net.classifier._modules.items():
        if isinstance(layer, nn.modules.linear.Linear):
            decomposed = svd_decomposition(layer, energy)
            net.classifier._modules[key] = decomposed

    print(net)
    checkpoint = {"model": net, "state_dict": net.state_dict()}
    torch.save(checkpoint, "Results/VGG_16_CP.PTH")
    print("without GPU:", timer() - start)

    import matlab.engine
    import torch
    from torch import nn
    from torchvision import models
    from ht2 import ht2
    from hosvd1 import hosvd1
    from svd_decomposition import svd_decomposition
    import matlab.engine
    from TuckerforVGG16 import tucker_decomposition_conv_layer
    from timeit import default_timer as timer
    from torch.profiler import profile, record_function, ProfilerActivity
    from ptflops import get_model_complexity_info
    from flopth import flopth
    import sys

    model_pretrained = models.vgg16_bn(True)
    #input_tensor = torch.rand(2, 4, 37, 37)
    input_tensor = torch.rand(3, 224, 224)
    input_tensor = torch.unsqueeze(input_tensor, 0)

    print("That's for the Decomposed VGG16:")
    flops1, params1 = flopth(net, in_size=((3, 224, 224),), show_detail=True)
    print(flops1, params1)


    print("Hi")


    net_pretrained = models.vgg16_bn(True)

    print("That's for the Decomposed VGG16:")
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            net(input_tensor)

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=None))

    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=None))

    import matplotlib.pyplot as plt
    import pandas as pd
    from flopco import FlopCo

    input_tensor = (1, 3, 224, 224)

    stats = FlopCo(net, img_size=input_tensor)

    print(stats.total_macs, stats.relative_flops)


    in_size = (1, 3, 224, 224)

    print("That's for the normal VGG16:")
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            net_pretrained(input_tensor)

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=None))

    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=None))


    print("That's for the Original VGG16:")
    with torch.autograd.profiler.profile() as prof:
        net_pretrained(input_tensor)

    flops_baseline_1 = 0
    flops_baseline_2 = 0
    for module in model_pretrained.modules():
        if isinstance(module, torch.nn.Conv2d):
            flops_baseline_1 += (
                    2 * module.weight.numel() * module.in_channels * module.out_channels * module.kernel_size[0] *
                    module.kernel_size[1] / module.groups)
        elif isinstance(module, torch.nn.Linear):
            flops_baseline_2 += (2 * module.weight.numel() * module.in_features)

    print(f'FLOPs for the original VGG16 model Conv2: {flops_baseline_1 / 1e9:.2f} billion')
    print(f'FLOPs for the original VGG16 model Linear: {flops_baseline_2 / 1e9:.2f} billion')

    print("That's for the decomposed VGG16:")
    with torch.autograd.profiler.profile() as prof:
        net(input_tensor)

    flops_baseline_3 = 0
    flops_baseline_4 = 0
    for module in net.modules():
        if isinstance(module, torch.nn.Conv2d):
            flops_baseline_3 += (
                    2 * module.weight.numel() * module.in_channels * module.out_channels * module.kernel_size[0] *
                    module.kernel_size[1] / module.groups)
        elif isinstance(module, torch.nn.Linear):
            flops_baseline_4 += (2 * module.weight.numel() * module.in_features)

    print(f'FLOPs for the decomposed VGG16 model Conv2: {flops_baseline_1 / 1e9:.2f} billion')
    print(f'FLOPs for the decomposed VGG16 model Linear: {flops_baseline_2 / 1e9:.2f} billion')



    print("That's for the normal VGG16:")
    flops1, params1 = flopth(net_pretrained, in_size=((3, 224, 224),), show_detail=True)

    print(flops1, params1)

    # Multiply Add operations

    print("That's for the decomposed VGG16:")

    macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity for decomposed VGG16: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters for decomposed VGG16: ', params))

    print("That's for the normal VGG16:")

    macs_pretrained, params_pretrained = get_model_complexity_info(net_pretrained, (3, 224, 224), as_strings=True,
                                                                   print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs_pretrained))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params_pretrained))

