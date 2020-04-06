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



class MobileNetV2(nn.Module):
    # Modify attributs	    
    def __init__(self, model):
        super(MobileNetV2, self).__init__()
        for key, val in model.__dict__.items():
            self.__dict__[key] = val

        self.g_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.last_linear = self.classifier

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
    return MobileNetV2(model)

@META_ARCH_REGISTRY.register()
def MobileNet_V2(cfg):
    return generate_model(cfg, 'mobilenet_v2')