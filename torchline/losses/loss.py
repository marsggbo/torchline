import torch
import torch.nn.functional as F
from .build import LOSS_FN_REGISTRY

__all__ = [
    'CrossEntropy'
]

@LOSS_FN_REGISTRY.register()
def CrossEntropy(cfg):
    return torch.nn.CrossEntropyLoss()
# class CrossEntropy(torch.nn.Module):
#     def __init__(self, cfg):
#         super(CrossEntropy, self).__init__()
#         self.loss_fn = F.cross_entropy
    
#     def forward(self, predictions, gt_labels):
#         return self.loss_fn(predictions, gt_labels, reduction="mean")
