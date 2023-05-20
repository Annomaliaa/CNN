# This is a sample Python script.
import numpy as np
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import torch
from torch import nn
from torchvision import models, transforms
from ht2 import ht2
from svd_decomposition import svd_decomposition
from hosvd1 import hosvd1
import sys
import matlab.engine
from torch.profiler import profile, record_function, ProfilerActivity
from ptflops import get_model_complexity_info
import csv
from ttblock import tt_block

from CP_DECOM import cp_decomposition_conv_layer
from flopth import flopth

from timeit import default_timer as timer
import pandas as pd
import tensorflow
from tt import tt
from TT_T3F import tt_2
from Tucker import normal_tucker
from TuckerforResnet import tucker_decomposition_conv_layer


#from deepspeed.profiling.flops_profiler import get_model_profile



sys.path.append("C:/Program Files/MATLAB/R2022b/bin/matlab.exe")
sys.path.append("C:/Users/Ania/Downloads/tensor_toolbox-v3.5")


if __name__ == "__main__":
    print(tensorflow.__version__)
    print(sys.version)
    print(torch.__version__)
    print(torch.cuda.is_available())
    start = timer()
    energy = 0.8
    net = models.resnet50(True)

    eng = matlab.engine.start_matlab()


    eng.addpath("C:/Program Files/MATLAB/R2022b/bin/matlab")
    eng.addpath("C:/Users/Ania/Documents/MATLAB/matlab")

    from Canonical2 import cp_decomposition_conv_layer

    layers = [net.layer1, net.layer2, net.layer3, net.layer4]



    decomposed = hosvd1(net.conv1, eng, energy)
    net.conv1 = decomposed

    for layer in layers:
        for block in layer:

            resnet50 = models.resnet50(pretrained=True)

            conv1 = block.conv1
            decomposed_conv1 = hosvd1(conv1, eng, energy)
            block.conv1 = decomposed_conv1

            conv2 = block.conv2
            decomposed_conv2 = cp_decomposition_conv_layer(conv2)
            #decomposed_conv2 = tucker_decomposition_conv_layer(conv2)
            block.conv2 = decomposed_conv2
            print(decomposed_conv2)

            conv3 = block.conv3
            decomposed_conv3 = hosvd1(conv3, eng, energy)
            block.conv3 = decomposed_conv3

            if block.downsample:
                conv = block.downsample[0]
                decomposed_conv = hosvd1(conv, eng, energy)
                block.downsample[0] = decomposed_conv

    for name, layer in net.named_modules():
        if isinstance(layer, nn.Conv2d):
            print(f'{name}: input shape: {layer.weight.shape}, output shape: {layer.out_channels}')

    fc = net.fc

    decomposed_fc = svd_decomposition(net.fc, energy)
    net.fc = decomposed_fc

    print(net)
    checkpoint = {"model": net, "state_dict": net.state_dict()}
    #torch.save(checkpoint, "Results/REESNET_50-Tucker-new-layers.PTH")

    net.eval()

    # create an input tensor
    input_tensor = torch.rand(1, 3, 224, 224)

    # pass the input tensor through the model
    output = net(input_tensor)

    # print the output tensor
    print(output)

    model_pretrained = models.resnet50(True)
    input_tensor = torch.rand(1, 3, 224, 224)

    print("without GPU:", timer() - start)

    print("That's for the Original ResNet50:")
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            out = torch.randn(1, 3, 224, 224)
            print(f"input shape: {out.shape}")
            model_pretrained(out)

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=None))

    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=None))

    import matplotlib.pyplot as plt
    import pandas as pd
    data = pd.read_csv('bar_cpu_normal.csv')

    df = pd.DataFrame(data)

    X = list(df.iloc[:, 0])
    Y = list(df.iloc[:, 1])

    fig = plt.figure(figsize=(30, 30))

    plt.bar(X, Y)
    plt.xticks(rotation=90)
    plt.xlabel('Layer')
    plt.ylabel('CPU Time (ms)')
    plt.show()

    print("That's for the Decomposed ResNet50:")
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            out = torch.randn(1, 3, 224, 224)
            print(f"input shape: {out.shape}")
            net(out)

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=None))

    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=None))

    print("That's for the Original ResNet50:")
    with torch.autograd.profiler.profile() as prof:
        model_pretrained(input_tensor)

    flops_baseline_1 = 0
    flops_baseline_2 = 0
    for module in model_pretrained.modules():
        if isinstance(module, torch.nn.Conv2d):
            flops_baseline_1 += (
                        2 * module.weight.numel() * module.in_channels * module.out_channels * module.kernel_size[0] *
                        module.kernel_size[1] / module.groups)
        elif isinstance(module, torch.nn.Linear):
            flops_baseline_2 += (2 * module.weight.numel() * module.in_features)

    print(f'FLOPs for the Original ResNet50 model Conv2: {flops_baseline_1 / 1e9:.2f} billion')
    print(f'FLOPs for the 0riginal ResNet50 model Linear: {flops_baseline_2 / 1e9:.2f} billion')

    print("That's for the Decomposed ResNet50:")
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

    print(f'FLOPs for the Decomposed ResNet50 model Conv2: {flops_baseline_1 / 1e9:.2f} billion')
    print(f'FLOPs for the Decomposed ResNet50 model Linear: {flops_baseline_2 / 1e9:.2f} billion')

    print("That's for the Decomposed ResNet50:")
    flops, params = flopth(net, in_size=((3, 224, 224),), show_detail=True)

    print(flops, params)

    print("That's for the Original ResNet50:")
    flops1, params1 = flopth(model_pretrained, in_size=((3, 224, 224),), show_detail=True)

    print(flops1, params1)

    # Multiply Add operations

    print("That's for the Decomposed ResNet50:")

    macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity for decomposed resnet50: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters for decomposed resnet50: ', params))

    print("That's for the Original ResNet50:")

    macs_pretrained, params_pretrained = get_model_complexity_info(model_pretrained, (3, 224, 224), as_strings=True,
                                                                   print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs_pretrained))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params_pretrained))

    flops_per_layer = []
    iteration_time_per_layer = []
    params_per_layer = []
    model = net
    input_data = torch.rand(1, 3, 224, 224)

    # Iterate through the model layers
    for i, layer in enumerate(model.modules()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            # Get the FLOPs and number of parameters for the layer
            flops, params = flopth(layer, in_size=((3, 224, 224),), show_detail=False)
            flops_per_layer.append(flops)
            params_per_layer.append(params)
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                _ = layer(torch.randn(64, 14, 1, 1))
            iteration_time = prof.key_averages().self_cpu_time_total / 1e6  # Convert to seconds
            iteration_time_per_layer.append(iteration_time)

    # Plot the data
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Iteration Time (ms)', color=color)
    ax1.plot(range(1, len(iteration_time_per_layer) + 1), iteration_time_per_layer, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # Create second y-axis

    color = 'tab:blue'
    ax2.set_ylabel('Flops', color=color)
    ax2.plot(range(1, len(flops_per_layer) + 1), flops_per_layer, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # Fit the plot nicely in the figure
    plt.title('Flops per Layer and Iteration Time')
    plt.show()

