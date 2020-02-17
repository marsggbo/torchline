import torch
import torch.nn.functional as F
from .build import LOSS_FN_REGISTRY

__all__ = [
    'CrossEntropy'
]

@LOSS_FN_REGISTRY.register()
def CrossEntropy(cfg):
    weight = cfg.loss.class_weight
    if weight in ['', None, []]:
        weight = None
    else:
        weight = torch.tensor(weight)
        if torch.cuda.is_available(): weight=weight.cuda()
    return torch.nn.CrossEntropyLoss(weight=weight)