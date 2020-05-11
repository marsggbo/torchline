
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
    '_DefaultTransforms',
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
    def __init__(self, is_train, log_name):
        self.logger_print = Logger(__name__, log_name).getlogger()
        self.is_train = is_train

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
def DefaultTransforms(cfg):
    is_train = cfg.dataset.is_train
    log_name = cfg.log.name
    mean = cfg.transforms.tensor.normalization.mean
    std = cfg.transforms.tensor.normalization.std
    img_size = cfg.input.size
    
    aug_imagenet = cfg.transforms.img.aug_imagenet
    aug_cifar = cfg.transforms.img.aug_cifar
    random_resized_crop = cfg.transforms.img.random_resized_crop
    resize = cfg.transforms.img.resize
    random_crop = cfg.transforms.img.random_crop
    center_crop = cfg.transforms.img.center_crop
    random_horizontal_flip = cfg.transforms.img.random_horizontal_flip
    random_vertical_flip = cfg.transforms.img.random_vertical_flip
    random_rotation = cfg.transforms.img.random_rotation
    color_jitter = cfg.transforms.img.color_jitter
    return _DefaultTransforms(is_train, log_name, img_size,
                 aug_imagenet, aug_cifar,
                 random_resized_crop,
                 resize,
                 random_crop,
                 center_crop,
                 random_horizontal_flip,
                 random_vertical_flip,
                 random_rotation,
                 color_jitter,
                 mean, std)

class _DefaultTransforms(BaseTransforms):
    def __init__(self, is_train, log_name, img_size,
                 aug_imagenet=False, aug_cifar=False,
                 random_resized_crop={'enable':0},
                 resize={'enable':1},
                 random_crop={'enable':0, 'padding':0},
                 center_crop={'enable':0},
                 random_horizontal_flip={'enbale':0, 'p':0.5},
                 random_vertical_flip={'enbale':0, 'p':0.5},
                 random_rotation={'enbale':0, 'degrees':15},
                 color_jitter={'enable':0,'brightness':0.1, 'contrast':0.1, 'saturation':0.1, 'hue':0.1},
                 mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], *args, **kwargs):
        super(_DefaultTransforms, self).__init__(is_train, log_name)
        self.is_train = is_train
        self.mean = mean
        self.std = std
        self.img_size = img_size
        self.min_edge_size = min(self.img_size)
        self.normalize = transforms.Normalize(self.mean, self.std)
        self.aug_imagenet = aug_imagenet
        self.aug_cifar = aug_cifar
        self.random_resized_crop = random_resized_crop
        self.resize = resize
        self.random_crop = random_crop
        self.center_crop = center_crop
        self.random_horizontal_flip = random_horizontal_flip
        self.random_vertical_flip = random_vertical_flip
        self.random_rotation = random_rotation
        self.color_jitter = color_jitter
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
        if self.aug_imagenet:
            self.logger_print.info('Using imagenet augmentation')
            transform = transforms.Compose([
                transforms.Resize(self.img_size),
                autoaugment.ImageNetPolicy(),
                transforms.ToTensor(),
                self.normalize
            ])
        # aug cifar
        elif self.aug_cifar:
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

        # resize and crop opertaion
        if self.random_resized_crop['enable']:
            transform_list.append(transforms.RandomResizedCrop(self.img_size))
        elif self.resize['enable']:
            transform_list.append(transforms.Resize(self.img_size))
        if self.random_crop['enable']:
            transform_list.append(transforms.RandomCrop(self.min_edge_size, padding=self.random_crop['padding']))
        elif self.center_crop['enable']:
            transform_list.append(transforms.CenterCrop(self.min_edge_size))

        # ColorJitter
        if self.color_jitter['enable']:
            params = {key: self.color_jitter[key] for key in self.color_jitter 
                            if key != 'enable'}
            transform_list.append(transforms.ColorJitter(**params))

        # horizontal flip
        if self.random_horizontal_flip['enable']:
            p = self.random_horizontal_flip['p']
            transform_list.append(transforms.RandomHorizontalFlip(p))
        
        # vertical flip
        if self.random_vertical_flip['enable']:
            p = self.random_vertical_flip['p']
            transform_list.append(transforms.RandomVerticalFlip(p))

        # rotation
        if self.random_rotation['enable']:
            degrees = self.random_rotation['degrees']
            transform_list.append(transforms.RandomRotation(degrees))
        transform_list.append(transforms.ToTensor())
        transform_list.append(self.normalize)
        transform_list = transforms.Compose(transform_list)
        assert len(transform_list.transforms) > 0, "You must apply transformations"
        return transform_list

    def check_conflict_options(self):
        count = self.random_resized_crop['enable'] + \
                self.resize['enable']
        assert count <= 1, 'You can only use one resize transform operation'

        count = self.random_crop['enable'] + \
                self.center_crop['enable']
        assert count <= 1, 'You can only use one crop transform operation'