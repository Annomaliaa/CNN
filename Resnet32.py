import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import dill
from flopth import flopth
from torch.profiler import profile, record_function, ProfilerActivity


__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']




def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':

                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4),
                                                  "constant", 0))


            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])

def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))

model = resnet32()
model = torch.nn.DataParallel(model)
param = torch.load('checkpoints/resnet32-d509ac18.th', map_location=torch.device('cpu'))
model.load_state_dict(param['state_dict'])

#torch.save(model, 'models/resnet_cifar/resnet32-pre-not-decomp.pth', pickle_module=dill)

input_tensor = (1, 3, 32, 32)
input = torch.rand(input_tensor)

print(model)

#out = model(input)

#print(out)

test(model)

model = model.module
layer1 = model.layer1
layer2 = model.layer2
layer3 = model.layer3


print(layer1)
print(layer2)
print(layer3)


layers = [layer1, layer2, layer3]

print(layers)

from timeit import default_timer as timer
from TuckerforResnet import tucker_decomposition_conv_layer
from ht2 import ht2
import matlab.engine
import sys

sys.path.append("C:/Program Files/MATLAB/R2022b/bin/matlab.exe")
sys.path.append("C:/Users/Ania/Downloads/tensor_toolbox-v3.5")

eng = matlab.engine.start_matlab()

eng.addpath("C:/Program Files/MATLAB/R2022b/bin/matlab")
eng.addpath("C:/Users/Ania/Documents/MATLAB/matlab")

energy = 0.8
start = timer()

from Canonical import cp_decomposition_conv_layer

for layer in layers:
    for block in layer:
        conv1 = block.conv1
        decomposed_conv1 = cp_decomposition_conv_layer(conv1)
        block.conv1 = decomposed_conv1
        print("Decomposed Layer Conv1", decomposed_conv1)
        conv2 = block.conv2
        decomposed_conv2 = cp_decomposition_conv_layer(conv2)
        block.conv2 = decomposed_conv2
        print("Decomposed Layer Conv2", decomposed_conv2)


print(model)

test(model)

input_tensor = torch.randn(1, 3, 224, 224)

print("That's for the Decomposed Resnet32:")
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(input_tensor)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=None))

print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=None))


checkpoint = {"model": model, "state_dict": model.state_dict()}
torch.save(checkpoint, 'Results/resnet32-decomp-cp.pth', pickle_module=dill)

print("That's for the Decomposed Resnet32:")
flops, params = flopth(model, in_size=((3, 224, 224),), show_detail=True)
print(flops, params)





