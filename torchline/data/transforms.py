
#coding=utf-8
import torch
import numpy as np
import math
import random
import torchvision
from torchvision import transforms
from . import autoaugment
from torchline.utils import Registry, Logger

TRANSFORMS_REGISTRY = Registry('transforms')
TRANSFORMS_REGISTRY.__doc__ = """
Registry for data transform functions, i.e. torchvision.transforms

The registered object will be called with `obj(cfg)`
"""

LABEL_TRANSFORMS_REGISTRY = Registry('label_transforms')
LABEL_TRANSFORMS_REGISTRY.__doc__ = """
Registry for label transform functions, i.e. torchvision.transforms

The registered object will be called with `obj(cfg)`
"""

__all__ = [
    'build_transforms',
    'build_label_transforms',
    'TRANSFORMS_REGISTRY',
    'LABEL_TRANSFORMS_REGISTRY',
    'DefaultTransforms',
    'BaseTransforms'
]


def build_transforms(cfg):
    """
    Built the transforms, defined by `cfg.transforms.name`.
    """
    name = cfg.transforms.name
    return TRANSFORMS_REGISTRY.get(name)(cfg)
    
def build_label_transforms(cfg):
    """
    Built the label transforms, defined by `cfg.label_transforms.name`.
    """
    name = cfg.label_transforms.name
    if name == 'default':
        return None
    return LABEL_TRANSFORMS_REGISTRY.get(name)(cfg)

class BaseTransforms(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger_print = Logger(__name__, cfg).getlogger()
        self.is_train = cfg.dataset.is_train
        
    def get_transform(self):
        if not self.is_train: 
            self.logger_print.info('Generating validation transform ...')
            transform = self.valid_transform
            self.logger_print.info(f'Valid transform={transform}')
        else:
            self.logger_print.info('Generating training transform ...')
            transform = self.train_transform
            self.logger_print.info(f'Train transform={transform}')
        return transform

    @property
    def valid_transform(self):
        raise NotImplementedError

    @property
    def train_transform(self):
        raise NotImplementedError


@TRANSFORMS_REGISTRY.register()
class DefaultTransforms(BaseTransforms):
    def __init__(self, cfg):
        super(DefaultTransforms, self).__init__(cfg)
        self.is_train = cfg.dataset.is_train
        self.mean = cfg.transforms.tensor.normalization.mean
        self.std = cfg.transforms.tensor.normalization.std
        self.img_size = cfg.input.size
        self.padding = cfg.transforms.img.random_crop.padding
        self.min_edge_size = min(self.img_size)
        self.normalize = transforms.Normalize(self.mean, self.std)
        self.transform = self.get_transform()

    @property
    def valid_transform(self):
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            self.normalize
        ])
        return transform

    @property
    def train_transform(self):
        # aug_imagenet
        if self.cfg.transforms.img.aug_imagenet:
            self.logger_print.info('Using imagenet augmentation')
            transform = transforms.Compose([
                transforms.Resize(self.img_size),
                autoaugment.ImageNetPolicy(),
                transforms.ToTensor(),
                self.normalize
            ])
        # aug cifar
        elif self.cfg.transforms.img.aug_cifar:
            self.logger_print.info('Using cifar augmentation')
            transform = transforms.Compose([
                transforms.Resize(self.img_size),
                autoaugment.CIFAR10Policy(),
                transforms.ToTensor(),
                self.normalize
            ])
        # customized transformations
        else:
            transform = self.read_transform_from_cfg()
        return transform

    def read_transform_from_cfg(self):
        transform_list = []
        self.check_conflict_options()
        img_transforms = self.cfg.transforms.img

        # resize and crop opertaion
        if img_transforms.random_resized_crop.enable:
            transform_list.append(transforms.RandomResizedCrop(self.img_size))
        elif img_transforms.resize.enable:
            transform_list.append(transforms.Resize(self.img_size))
        if img_transforms.random_crop.enable:
            transform_list.append(transforms.RandomCrop(self.min_edge_size, padding=self.padding))
        elif img_transforms.center_crop.enable:
            transform_list.append(transforms.CenterCrop(self.min_edge_size))

        # ColorJitter
        if img_transforms.color_jitter.enable:
            params = {key: img_transforms.color_jitter[key] for key in img_transforms.color_jitter 
                            if key != 'enable'}
            transform_list.append(transforms.ColorJitter(**params))

        # horizontal flip
        if img_transforms.random_horizontal_flip.enable:
            p = img_transforms.random_horizontal_flip.p
            transform_list.append(transforms.RandomHorizontalFlip(p))
        
        # vertical flip
        if img_transforms.random_vertical_flip.enable:
            p = img_transforms.random_vertical_flip.p
            transform_list.append(transforms.RandomVerticalFlip(p))

        # rotation
        if img_transforms.random_rotation.enable:
            degrees = img_transforms.random_rotation.degrees
            transform_list.append(transforms.RandomRotation(degrees))
        transform_list.append(transforms.ToTensor())
        transform_list.append(self.normalize)
        transform_list = transforms.Compose(transform_list)
        assert len(transform_list.transforms) > 0, "You must apply transformations"
        return transform_list

    def check_conflict_options(self):
        count = self.cfg.transforms.img.random_resized_crop.enable + \
                self.cfg.transforms.img.resize.enable
        assert count <= 1, 'You can only use one resize transform operation'

        count = self.cfg.transforms.img.random_crop.enable + \
                self.cfg.transforms.img.center_crop.enable
        assert count <= 1, 'You can only use one crop transform operation'