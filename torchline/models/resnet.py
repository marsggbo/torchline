'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from .build import META_ARCH_REGISTRY

__all__ = [
    'ResNet',
    'Resnet18',
    'Resnet34',
    'Resnet50',
    'Resnet101',
    'Resnet152',
]

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x): # 1*3*32*32
        
        out = F.relu(self.bn1(self.conv1(x))) # 1*64*32*32
        out = self.layer1(out) # 1*256*32*32
        out = self.layer2(out) # 1*512*16*16
        out = self.layer3(out) # 1*1024*8*8
        out = self.layer4(out) # 1*2048*4*4
        out = nn.AdaptiveAvgPool2d((1,1))(out) # 1*2048*1*1
        out = out.view(out.size(0), -1) # 1*2048
        out = self.linear(out) # 1*10
        return out


@META_ARCH_REGISTRY.register()
class Resnet18(ResNet):
    def __init__(self, cfg):
        num_classes = cfg.model.classes
        super(Resnet18, self).__init__(BasicBlock, [2,2,2,2], num_classes=num_classes)
    

@META_ARCH_REGISTRY.register()
class Resnet34(ResNet):
    def __init__(self, cfg):
        num_classes = cfg.model.classes
        super(Resnet34, self).__init__(BasicBlock, [3,4,6,3], num_classes=num_classes)

@META_ARCH_REGISTRY.register()
class Resnet50(ResNet):
    def __init__(self, cfg):
        num_classes = cfg.model.classes
        super(Resnet50, self).__init__(Bottleneck, [3,4,6,3], num_classes=num_classes)

@META_ARCH_REGISTRY.register()
class Resnet101(ResNet):
    def __init__(self, cfg):
        num_classes = cfg.model.classes
        super(Resnet101, self).__init__(Bottleneck, [3,4,23,3], num_classes=num_classes)

@META_ARCH_REGISTRY.register()
class Resnet152(ResNet):
    def __init__(self, cfg):
        num_classes = cfg.model.classes
        super(Resnet152, self).__init__(Bottleneck, [3,8,36,3], num_classes=num_classes)


def test():
    from torchline.config import get_cfg
    from torchline.models import build_model
    cfg = get_cfg()
    cfg.model.name = 'Resnet50'
    cfg.model.classes = 10
    net = build_model(cfg)
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
