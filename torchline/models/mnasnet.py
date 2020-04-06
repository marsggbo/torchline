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

class MNASNet(nn.Module):
    # Modify attributs	    
    def __init__(self, model):
        super(MNASNet, self).__init__()
        for key, val in model.__dict__.items():
            self.__dict__[key] = val
        self.stem   = model.layers[:8]
        self.layer1 = model.layers[8]
        self.layer2 = model.layers[9]
        self.layer3 = model.layers[10]
        self.layer4 = model.layers[11]
        self.layer5 = model.layers[12]
        self.layer6 = model.layers[13]
        self.layer7 = model.layers[14:]
        self.g_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.last_linear = model.classifier

    def features(self, x): # b*3*64*64
        out = self.stem(x)   # b*16*32*32
        out = self.layer1(out)     # b*16*16*16
        out = self.layer2(out)     # b*24*8*8
        out = self.layer3(out)     # b*40*4*4
        out = self.layer4(out)     # b*48*4*4
        out = self.layer5(out)     # b*96*2*2
        out = self.layer6(out)     # b*160*2*2
        out = self.layer7(out)     # b*1280*2*2
        return out

    def logits(self, x):
        out = x.mean([2, 3])
        out = self.last_linear(out)
        return out

    def forward(self, x):
        out = self.features(x)
        out = self.logits(out)
        return out

def generate_model(cfg, name):
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
    return MNASNet(model)

@META_ARCH_REGISTRY.register()
def MNASNet0_5(cfg):
    return generate_model(cfg, 'mnasnet0_5')
    
@META_ARCH_REGISTRY.register()
def MNASNet0_75(cfg):
    return generate_model(cfg, 'mnasnet0_75')
    
@META_ARCH_REGISTRY.register()
def MNASNet1_0(cfg):
    return generate_model(cfg, 'mnasnet1_0')
    
@META_ARCH_REGISTRY.register()
def MNASNet1_3(cfg):
    return generate_model(cfg, 'mnasnet1_3')