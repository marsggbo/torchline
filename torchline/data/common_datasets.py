import torch
from torchvision.datasets import CIFAR10 as _CIFAR10
from torchvision.datasets import MNIST as _MNIST

from .build import DATASET_REGISTRY
from .transforms import build_transforms

__all__ = [
    'MNIST',
    'CIFAR10',
    'FakeData',
    'fakedata'
]

@DATASET_REGISTRY.register()
def MNIST(cfg):
    root = cfg.dataset.dir
    is_train = cfg.dataset.is_train
    transform = build_transforms(cfg)
    return _MNIST(root=root, train=is_train, transform=transform.transform, download=True)

@DATASET_REGISTRY.register()
def CIFAR10(cfg):
    root = cfg.dataset.dir
    is_train = cfg.dataset.is_train
    transform = build_transforms(cfg)
    return _CIFAR10(root=root, train=is_train, transform=transform.transform, download=True)

class FakeData(torch.utils.data.Dataset):
    def __init__(self, size=64, num=100):
        if isinstance(size, int):
            self.size = [size, size]
        elif isinstance(size, list):
            self.size = size
        self.num = num
        self.data = torch.rand(num, 3, *size)
        self.labels = torch.randint(0, 10, (num,))


    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.num

@DATASET_REGISTRY.register()
def fakedata(cfg):
    size = cfg.input.size
    return FakeData(size)