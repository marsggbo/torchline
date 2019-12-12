import torch
import torch.nn.functional as F
from .build import LOSS_FN_REGISTRY

__all__ = [
    'cross_entropy'
]

@LOSS_FN_REGISTRY.register()
class CrossEntropy:
    def __init__(self, cfg):
        self.loss_fn = F.cross_entropy
    
    def __call__(self, predictions, gt_labels):
        return self.loss_fn(predictions, gt_labels, reduction="mean")
