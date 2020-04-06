import types

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import ResNet

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


class ResNet(nn.Module):
    # Modify attributs	    
    def __init__(self, model):
        super(ResNet, self).__init__()
        for key, val in model.__dict__.items():
            self.__dict__[key] = val
        self.g_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.last_linear = self.fc

    def features(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

    def logits(self, x):
        out = self.g_avg_pool(x)
        out = out.view(out.size(0), -1)
        out = self.last_linear(out)
        return out

    def forward(self, x):
        out = self.features(x)
        out = self.logits(out)
        return out

def generate_model(cfg, name):
    pretrained=cfg.model.pretrained
    classes = cfg.model.classes
    model = eval(f"models.{name}(pretrained={pretrained})")
    if classes != 1000:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, classes, bias=False)
    return ResNet(model)

@META_ARCH_REGISTRY.register()
def Resnet18(cfg):
    return generate_model(cfg, 'resnet18')

@META_ARCH_REGISTRY.register()
def Resnet34(cfg):
    return generate_model(cfg, 'resnet34')

@META_ARCH_REGISTRY.register()
def Resnet50(cfg):
    return generate_model(cfg, 'resnet50')

@META_ARCH_REGISTRY.register()
def Resnet101(cfg):
    return generate_model(cfg, 'resnet101')

@META_ARCH_REGISTRY.register()
def Resnet152(cfg):
    return generate_model(cfg, 'resnet152')

@META_ARCH_REGISTRY.register()
def Resnext50_32x4d(cfg):
    return generate_model(cfg, 'resnext50_32x4d')

@META_ARCH_REGISTRY.register()
def Resnext101_32x8d(cfg):
    return generate_model(cfg, 'resnext101_32x8d')

@META_ARCH_REGISTRY.register()
def Wide_resnet50_2(cfg):
    return generate_model(cfg, 'wide_resnet50_2')

@META_ARCH_REGISTRY.register()
def Wide_resnet101_2(cfg):
    return generate_model(cfg, 'wide_resnet101_2')