
import torch
from torchvision import models
from TuckerforResnet import tucker_decomposition_conv_layer
from timeit import default_timer as timer
import sys


if __name__ == "__main__":

    net = models.vgg16_bn(True)

    start = timer()


    for i, key in enumerate(net.features._modules.keys()):

        if isinstance(net.features._modules[key], torch.nn.modules.conv.Conv2d):
            #decomposed_tucker = tucker_decomposition_conv_layer(layer)
            conv_layer = net.features._modules[key]
            decomposed_cp = tucker_decomposition_conv_layer(conv_layer)
            net.features._modules[key] = decomposed_cp
            print("Decomposed Layer", decomposed_cp)
            original_weights = conv_layer.weight.cpu().data.numpy()
            decomposed_weights = decomposed_cp[1].weight.data.cpu().numpy()


    print(net)
    checkpoint = {"model": net, "state_dict": net.state_dict()}
    torch.save(checkpoint, "Results/VGG_16_CP.PTH")
    print("without GPU:", timer() - start)

    import torch
    from torch import nn
    from torchvision import models
    from TuckerforResnet import tucker_decomposition_conv_layer
    from timeit import default_timer as timer
    from torch.profiler import profile, record_function, ProfilerActivity
    from ptflops import get_model_complexity_info
    from flopth import flopth



    model_pretrained = models.vgg16_bn(True)
    #input_tensor = torch.rand(2, 4, 37, 37)
    input_tensor = torch.rand(3, 224, 224)
    input_tensor = torch.unsqueeze(input_tensor, 0)

    print("That's for the decomposed VGG16:")
    flops1, params1 = flopth(net, in_size=((3, 224, 224),), show_detail=True)
    print(flops1, params1)


    print("Hi")


    net_pretrained = models.vgg16_bn(True)

    print("That's for the decomposed VGG16:")
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            net(input_tensor)

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=None))

    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=None))

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


    print("That's for the normal VGG16:")
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

