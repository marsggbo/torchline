import torch
import torch.nn.functional as F
from .build import LOSS_FN_REGISTRY

__all__ = [
    'CrossEntropy'
]

@LOSS_FN_REGISTRY.register()
def CrossEntropy(cfg):
    weight = cfg.LOSS.CLASS_WEIGHT
    if not weight:
        weight = None
    return torch.nn.CrossEntropyLoss(weight=weight)