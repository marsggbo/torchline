#coding=utf-8
import torch
import torch.nn.functional as F
from .build import LOSS_FN_REGISTRY

__all__ = [
    'CrossEntropy',
    'CrossEntropyLabelSmooth',
    '_CrossEntropyLabelSmooth'
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

@LOSS_FN_REGISTRY.register()
def CrossEntropyLabelSmooth(cfg):
    try:
        label_smoothing = cfg.loss.label_smoothing
    except:
        label_smoothing = 0.1
    return _CrossEntropyLabelSmooth(label_smoothing)

class _CrossEntropyLabelSmooth(torch.nn.Module):
    def __init__(self, label_smoothing):
        super(_CrossEntropyLabelSmooth, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, pred, target):
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        n_classes = pred.size(1)
        # convert to one-hot
        target = torch.unsqueeze(target, 1)
        soft_target = torch.zeros_like(pred)
        soft_target.scatter_(1, target, 1)
        # label smoothing
        soft_target = soft_target * (1 - self.label_smoothing) + self.label_smoothing / n_classes
        return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))