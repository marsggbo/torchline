from torchvision.datasets import MNIST as _MNIST
from torchvision.datasets import CIFAR10 as _CIFAR10
from .build import DATASET_REGISTRY
from .transforms import build_transforms


__all__ = [
    'MNIST',
    'CIFAR10',
]

@DATASET_REGISTRY.register()
def MNIST(cfg):
    root = cfg.DATASET.DIR
    is_train = cfg.DATASET.IS_TRAIN
    transform = build_transforms(cfg)
    return _MNIST(root=root, train=is_train, transform=transform.transform, download=True)

@DATASET_REGISTRY.register()
def CIFAR10(cfg):
    root = cfg.DATASET.DIR
    is_train = cfg.DATASET.IS_TRAIN
    transform = build_transforms(cfg)
    return _CIFAR10(root=root, train=is_train, transform=transform.transform, download=True)