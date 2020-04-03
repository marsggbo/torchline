import types

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.mobilenet import MobileNetV2

from .build import META_ARCH_REGISTRY

__all__ = [
    'MobileNetV2',
    'MobileNet_V2'
]


def modify_mobilenet(model):
    # Modify attributs
    model.last_linear = model.classifier
    model.classifier = None

    def logits(self, features):
        x = features.mean([2, 3])
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model

def generate_mobilenet(cfg, name):
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
    return modify_mobilenet(model)

@META_ARCH_REGISTRY.register()
def MobileNet_V2(cfg):
    return generate_mobilenet(cfg, 'mobilenet_v2')