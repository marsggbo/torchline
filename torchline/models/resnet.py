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


def modify_resnets(model):
    # Modify attributs
    model.last_linear = model.fc
    model.fc = None

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, features):
        x = self.avgpool(features)
        x = x.view(x.size(0), -1)
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

def generate_resnet(cfg, name):
    pretrained=cfg.model.pretrained
    classes = cfg.model.classes
    model = eval(f"models.{name}(pretrained={pretrained})")
    if classes != 1000:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, classes, bias=False)
    return modify_resnets(model)

@META_ARCH_REGISTRY.register()
def Resnet18(cfg):
    return generate_resnet(cfg, 'resnet18')

@META_ARCH_REGISTRY.register()
def Resnet34(cfg):
    return generate_resnet(cfg, 'resnet34')

@META_ARCH_REGISTRY.register()
def Resnet50(cfg):
    return generate_resnet(cfg, 'resnet50')

@META_ARCH_REGISTRY.register()
def Resnet101(cfg):
    return generate_resnet(cfg, 'resnet101')

@META_ARCH_REGISTRY.register()
def Resnet152(cfg):
    return generate_resnet(cfg, 'resnet152')

@META_ARCH_REGISTRY.register()
def Resnext50_32x4d(cfg):
    return generate_resnet(cfg, 'resnext50_32x4d')

@META_ARCH_REGISTRY.register()
def Resnext101_32x8d(cfg):
    return generate_resnet(cfg, 'resnext101_32x8d')

@META_ARCH_REGISTRY.register()
def Wide_resnet50_2(cfg):
    return generate_resnet(cfg, 'wide_resnet50_2')

@META_ARCH_REGISTRY.register()
def Wide_resnet101_2(cfg):
    return generate_resnet(cfg, 'wide_resnet101_2')


# demo: customize the forward process
def customize():
    import type
    res = models.resnet18(pretrained=False)
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
        print("Hello")
        return x
    res.forward = forward
    x = torch.rand(1,3,64,64)
    y = res(res, x)
    print(y.shape)