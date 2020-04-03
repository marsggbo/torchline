'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.resnet import BasicBlock, Bottleneck

from .build import META_ARCH_REGISTRY

__all__ = [
    'ResNet',
    'Resnet18',
    'Resnet34',
    'Resnet50',
    'Resnet101',
    'Resnet152',
    'Resnext50_32x4d',
    'Resnext101_32x8d',
    'Wide_resnet50_2',
    'Wide_resnet101_2'
]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

class ResNet(models.ResNet):
    def __init__(self, block, num_blocks, classes=10, **kwargs):
        super(ResNet, self).__init__(block, num_blocks, **kwargs)

    def load_pretrained_weights(self, name, expansion, classes, pretrained=False):
        '''pretrained
        Args:
            name: model name, e.g. 'resnet18'
            expansion: block expansion
            classes: the number of classes
        '''
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls[name], progress=True)
            self.load_state_dict(state_dict)
        if classes != 1000:
            self.fc = nn.Linear(512*expansion, classes, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

@META_ARCH_REGISTRY.register()
class Resnet18(ResNet):
    def __init__(self, cfg):
        classes = cfg.model.classes
        super(Resnet18, self).__init__(BasicBlock, [2,2,2,2], classes=classes)
        self.load_pretrained_weights('resnet18', BasicBlock.expansion, classes, cfg.model.pretrained)
        
@META_ARCH_REGISTRY.register()
class Resnet34(ResNet):
    def __init__(self, cfg):
        classes = cfg.model.classes
        super(Resnet34, self).__init__(BasicBlock, [3,4,6,3], classes=classes)
        self.load_pretrained_weights('resnet50', BasicBlock.expansion, classes, cfg.model.pretrained)

@META_ARCH_REGISTRY.register()
class Resnet50(ResNet):
    def __init__(self, cfg):
        classes = cfg.model.classes
        super(Resnet50, self).__init__(Bottleneck, [3,4,6,3], classes=classes)
        self.load_pretrained_weights('resnet50', Bottleneck.expansion, classes, cfg.model.pretrained)

@META_ARCH_REGISTRY.register()
class Resnet101(ResNet):
    def __init__(self, cfg):
        classes = cfg.model.classes
        super(Resnet101, self).__init__(Bottleneck, [3,4,23,3], classes=classes)
        self.load_pretrained_weights('resnet101', Bottleneck.expansion, classes, cfg.model.pretrained)

@META_ARCH_REGISTRY.register()
class Resnet152(ResNet):
    def __init__(self, cfg):
        classes = cfg.model.classes
        super(Resnet152, self).__init__(Bottleneck, [3,8,36,3], classes=classes)
        self.load_pretrained_weights('resnet152', Bottleneck.expansion, classes, cfg.model.pretrained)


@META_ARCH_REGISTRY.register()
class Resnext50_32x4d(ResNet):
    def __init__(self, cfg):
        classes = cfg.model.classes
        super(Resnext50_32x4d, self).__init__(Bottleneck, [3, 4, 6, 3], classes=classes, groups=32, width_per_group=4)
        self.load_pretrained_weights('resnext50_32x4d', Bottleneck.expansion, classes, cfg.model.pretrained)

@META_ARCH_REGISTRY.register()
class Resnext101_32x8d(ResNet):
    def __init__(self, cfg):
        classes = cfg.model.classes
        super(Resnext101_32x8d, self).__init__(Bottleneck, [3, 4, 23, 3], classes=classes, groups=32, width_per_group=8)
        self.load_pretrained_weights('resnext101_32x8d', Bottleneck.expansion, classes, cfg.model.pretrained)

@META_ARCH_REGISTRY.register()
class Wide_resnet50_2(ResNet):
    def __init__(self, cfg):
        classes = cfg.model.classes
        super(Wide_resnet50_2, self).__init__(Bottleneck, [3, 4, 6, 3], classes=classes, width_per_group=64 * 2)
        self.load_pretrained_weights('wide_resnet50_2', Bottleneck.expansion, classes, cfg.model.pretrained)

@META_ARCH_REGISTRY.register()
class Wide_resnet101_2(ResNet):
    def __init__(self, cfg):
        classes = cfg.model.classes
        super(Wide_resnet101_2, self).__init__(Bottleneck, [3, 4, 23, 3], classes=classes, width_per_group=64 * 2)
        self.load_pretrained_weights('wide_resnet101_2', Bottleneck.expansion, classes, cfg.model.pretrained)


# demo: customize the forward process
# class Net(Resnet18):
#     def __init__(self, cfg):
#         super(Net, self).__init__(cfg)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         print("Hello")
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         return x

def test():
    import sys
    sys.path.append('..')
    from torchline.config import get_cfg
    from torchline.models import build_model
    cfg = get_cfg()
    cfg.model.name = 'Resnet18'
    cfg.model.pretrained = False
    cfg.model.classes = 10
    print(cfg.model)
    net = build_model(cfg)
    y = net(torch.randn(1,3,64,64))
    print(y.size())

if __name__ == '__main__':
    test()
