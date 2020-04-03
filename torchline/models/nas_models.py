import pretrainedmodels
import torch.nn as nn

from torchline.models import META_ARCH_REGISTRY

__all__ = [
    'Nasnetamobile',
    'Pnasnet5large',
]

def generate_model(cfg, name):
    pretrained='imagenet' if cfg.model.pretrained else None
    classes = cfg.model.classes
    img_size = cfg.input.size[0]
    model = pretrainedmodels.__dict__[name](num_classes=1000, pretrained=pretrained)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    if classes != 1000:
        in_features = model.last_linear.in_features
        model.last_linear = nn.Sequential(
            nn.Linear(in_features, classes, bias=False))
    return model

@META_ARCH_REGISTRY.register()
def Nasnetamobile(cfg):
    return generate_model(cfg, 'nasnetamobile')

@META_ARCH_REGISTRY.register()
def Pnasnet5large(cfg):
    return generate_model(cfg, 'pnasnet5large')
