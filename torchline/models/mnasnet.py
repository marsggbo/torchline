import types

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.mnasnet import MNASNet

from .build import META_ARCH_REGISTRY

__all__ = [
    'MNASNet',
    'MNASNet0_5',
    'MNASNet0_75',
    'MNASNet1_0',
    'MNASNet1_3',
]

# layers = [
#     # First layer: regular conv.
#     nn.Conv2d(3, depths[0], 3, padding=1, stride=2, bias=False),
#     nn.BatchNorm2d(depths[0], momentum=_BN_MOMENTUM),
#     nn.ReLU(inplace=True),
#     # Depthwise separable, no skip.
#     nn.Conv2d(depths[0], depths[0], 3, padding=1, stride=1,
#                 groups=depths[0], bias=False),
#     nn.BatchNorm2d(depths[0], momentum=_BN_MOMENTUM),
#     nn.ReLU(inplace=True),
#     nn.Conv2d(depths[0], depths[1], 1, padding=0, stride=1, bias=False),
#     nn.BatchNorm2d(depths[1], momentum=_BN_MOMENTUM),
#     # MNASNet blocks: stacks of inverted residuals.
#     _stack(depths[1], depths[2], 3, 2, 3, 3, _BN_MOMENTUM), #layer1
#     _stack(depths[2], depths[3], 5, 2, 3, 3, _BN_MOMENTUM), #layer2
#     _stack(depths[3], depths[4], 5, 2, 6, 3, _BN_MOMENTUM), #layer3
#     _stack(depths[4], depths[5], 3, 1, 6, 2, _BN_MOMENTUM), #layer4
#     _stack(depths[5], depths[6], 5, 2, 6, 4, _BN_MOMENTUM), #layer5
#     _stack(depths[6], depths[7], 3, 1, 6, 1, _BN_MOMENTUM), #layer6
#     # Final mapping to classifier input.
#     nn.Conv2d(depths[7], 1280, 1, padding=0, stride=1, bias=False),
#     nn.BatchNorm2d(1280, momentum=_BN_MOMENTUM),
#     nn.ReLU(inplace=True),
# ]
def modify_mnasnet(model):
    # Modify attributs
    model.stem = model.layers[:8]
    model.layer1 = model.layers[8]
    model.layer2 = model.layers[9]
    model.layer3 = model.layers[10]
    model.layer4 = model.layers[11]
    model.layer5 = model.layers[12]
    model.layer6 = model.layers[13]
    model.layer7 = model.layers[14:]
    model.last_linear = model.classifier
    model.layers = None
    model.classifier = None

    def features(self, input): # b*3*64*64
        x = self.stem(input)   # b*16*32*32
        x = self.layer1(x)     # b*16*16*16
        x = self.layer2(x)     # b*24*8*8
        x = self.layer3(x)     # b*40*4*4
        x = self.layer4(x)     # b*48*4*4
        x = self.layer5(x)     # b*96*2*2
        x = self.layer6(x)     # b*160*2*2
        x = self.layer7(x)     # b*1280*2*2
        return x

    def logits(self, features):
        x = features.mean([2, 3])
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model

def generate_mnasnet(cfg, name):
    pretrained=cfg.model.pretrained
    classes = cfg.model.classes
    if 'dropout' in cfg.model:
        dropout = cfg.model.dropout
    else:
        dropout = 0.2
    model = eval(f"models.{name}(pretrained={pretrained})")
    if classes != 1000:
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, classes, bias=False))
    return modify_mnasnet(model)

@META_ARCH_REGISTRY.register()
def MNASNet0_5(cfg):
    return generate_mnasnet(cfg, 'mnasnet0_5')
    
@META_ARCH_REGISTRY.register()
def MNASNet0_75(cfg):
    return generate_mnasnet(cfg, 'mnasnet0_75')
    
@META_ARCH_REGISTRY.register()
def MNASNet1_0(cfg):
    return generate_mnasnet(cfg, 'mnasnet1_0')
    
@META_ARCH_REGISTRY.register()
def MNASNet1_3(cfg):
    return generate_mnasnet(cfg, 'mnasnet1_3')