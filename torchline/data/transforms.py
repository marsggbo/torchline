
#coding=utf-8
import torch
import numpy as np
import math
import random
import torchvision
from torchvision import transforms
from . import autoaugment
from .data_utils import Cutout, RandomErasing
from torchline.utils import Registry, Logger

TRANSFORMS_REGISTRY = Registry('TRANSFORMS')
TRANSFORMS_REGISTRY.__doc__ = """
Registry for data transform functions, i.e. torchvision.transforms

The registered object will be called with `obj(cfg)`
"""

LABEL_TRANSFORMS_REGISTRY = Registry('LABEL_TRANSFORMS')
LABEL_TRANSFORMS_REGISTRY.__doc__ = """
Registry for label transform functions, i.e. torchvision.transforms

The registered object will be called with `obj(cfg)`
"""

__all__ = [
    'build_transforms',
    'build_label_transforms',
    'TRANSFORMS_REGISTRY',
    'LABEL_TRANSFORMS_REGISTRY',
    'DefaultTransforms'
]

logger_print = Logger(__name__).getlog()

def build_transforms(cfg):
    """
    Built the transforms, defined by `cfg.TRANSFORMS.NAME`.
    """
    name = cfg.TRANSFORMS.NAME
    return TRANSFORMS_REGISTRY.get(name)(cfg)
    
def build_label_transforms(cfg):
    """
    Built the label transforms, defined by `cfg.LABEL_TRANSFORMS.NAME`.
    """
    name = cfg.LABEL_TRANSFORMS.NAME
    if name == 'default':
        return None
    return LABEL_TRANSFORMS_REGISTRY.get(name)(cfg)

@TRANSFORMS_REGISTRY.register()
class DefaultTransforms:
    def __init__(self, cfg):
        self.cfg = cfg
        self.is_train = cfg.DATASET.IS_TRAIN
        self.mean = cfg.TRANSFORMS.TENSOR.NORMALIZATION.mean
        self.std = cfg.TRANSFORMS.TENSOR.NORMALIZATION.std
        self.img_size = cfg.INPUT.SIZE
        self.padding = cfg.TRANSFORMS.IMG.RANDOM_CROP.padding
        self.min_edge_size = min(self.img_size)
        self.normalize = transforms.Normalize(self.mean, self.std)
        self.transform = self.get_transform()

    def get_transform(self):
        # validation transform
        if not self.is_train: 
            logger_print.info('Generating validation transform ...')
            transform = self.valid_transform
            logger_print.info(f'Valid transform={transform}')
        else:
            logger_print.info('Generating training transform ...')
            transform = self.train_transform
            logger_print.info(f'Train transform={transform}')
        return transform


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
        if self.cfg.TRANSFORMS.IMG.AUG_IMAGENET:
            logger_print.info('Using imagenet augmentation')
            transform = transforms.Compose([
                transforms.Resize(self.min_edge_size+1),
                transforms.RandomCrop(self.min_edge_size, padding=self.padding),
                autoaugment.ImageNetPolicy(),
                transforms.ToTensor(),
                self.normalize
            ])
        # aug cifar
        elif self.cfg.TRANSFORMS.IMG.AUG_CIFAR:
            logger_print.info('Using cifar augmentation')
            transform = transforms.Compose([
                transforms.Resize(self.min_edge_size+1),
                transforms.RandomCrop(self.min_edge_size, padding=self.padding),
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
        img_transforms = self.cfg.TRANSFORMS.IMG

        # resize and crop opertaion
        if img_transforms.RANDOM_RESIZED_CROP.enable:
            transform_list.append(transforms.RandomResizedCrop(self.img_size))
        elif img_transforms.RESIZE.enable:
            transform_list.append(transforms.Resize(self.min_edge_size))
        if img_transforms.RANDOM_CROP.enable:
            transform_list.append(transforms.RandomCrop(self.min_edge_size, padding=self.padding))
        elif img_transforms.CENTER_CROP.enable:
            transform_list.append(transforms.CenterCrop(self.min_edge_size))

        # ColorJitter
        if img_transforms.COLOR_JITTER.enable:
            params = {key: img_transforms.COLOR_JITTER[key] for key in img_transforms.COLOR_JITTER 
                            if key != 'enable'}
            transform_list.append(transforms.ColorJitter(**params))

        # horizontal flip
        if img_transforms.RANDOM_HORIZONTAL_FLIP.enable:
            p = img_transforms.RANDOM_HORIZONTAL_FLIP.p
            transform_list.append(transforms.RandomHorizontalFlip(p))
        
        # vertical flip
        if img_transforms.RANDOM_VERTICAL_FLIP.enable:
            p = img_transforms.RANDOM_VERTICAL_FLIP.p
            transform_list.append(transforms.RandomVerticalFlip(p))

        # rotation
        if img_transforms.RANDOM_ROTATION.enable:
            degrees = img_transforms.RANDOM_ROTATION.degrees
            transform_list.append(transforms.RandomRotation(degrees))
        transform_list.append(transforms.ToTensor())
        transform_list.append(self.normalize)
        transform_list = transforms.Compose(transform_list)
        assert len(transform_list.transforms) > 0, "You must apply transformations"
        return transform_list

    def check_conflict_options(self):
        count = self.cfg.TRANSFORMS.IMG.RANDOM_RESIZED_CROP.enable + \
                self.cfg.TRANSFORMS.IMG.RESIZE.enable
        assert count <= 1, 'You can only use one resize transform operation'

        count = self.cfg.TRANSFORMS.IMG.RANDOM_CROP.enable + \
                self.cfg.TRANSFORMS.IMG.CENTER_CROP.enable
        assert count <= 1, 'You can only use one crop transform operation'