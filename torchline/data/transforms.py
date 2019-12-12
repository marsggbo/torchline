
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

__all__ = [
    'DefaultTransforms'
]

TRANSFORMS_REGISTRY = Registry('Transforms')
TRANSFORMS_REGISTRY.__doc__ = """

"""

logger = Logger(__name__).getlog()

def build_transforms(cfg):
    """
    Built the transforms, defined by `cfg.TRANSFORMS.NAME`.
    """
    name = cfg.TRANSFORMS.NAME
    return TRANSFORMS_REGISTRY.get(name)(cfg)

@TRANSFORMS_REGISTRY.register()
class DefaultTransforms:
    def __init__(self, cfg):
        self.cfg = cfg
        self.is_train = cfg.DATASET.IS_TRAIN
        self.mean = cfg.TRANSFORMS.TENSOR.NORMALIZATION.mean
        self.std = cfg.TRANSFORMS.TENSOR.NORMALIZATION.std
        self.img_size = cfg.INPUT.SIZE
        self.min_edge_size = min(self.img_size)
        self.normalize = transforms.Normalize(self.mean, self.std)
        self.transform = self.get_transform()

    def get_transform(self):
        # validation transform
        if not self.is_train: 
            logger.info('Generating validation transform ...')
            transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                self.normalize
            ])
            return transform

        logger.info('Generating training transform ...')
        if self.cfg.TRANSFORMS.IMG.AUG_IMAGENET:
            logger.info('Using imagenet augmentation')
            transform = transforms.Compose([
                transforms.Resize(self.min_edge_size+1),
                transforms.RandomCrop(self.min_edge_size),
                autoaugment.ImageNetPolicy(),
                transforms.ToTensor(),
                self.normalize
            ])
        elif self.cfg.TRANSFORMS.IMG.AUG_CIFAR:
            logger.info('Using cifar augmentation')
            transform = transforms.Compose([
                transforms.Resize(self.min_edge_size+1),
                transforms.RandomCrop(self.min_edge_size),
                autoaugment.CIFAR10Policy(),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            transform = self.read_transform_from_cfg()
        return transform

    def read_transform_from_cfg(self):
        transform_list = []
        self.check_resize_options()
        img_transforms = self.cfg.TRANSFORMS.IMG

        # resize and crop opertaion
        if img_transforms.RANDOM_RESIZED_CROP.enable:
            transform_list.append(transforms.RandomResizedCrop(self.img_size))
        elif img_transforms.RESIZE.enable:
            transform_list.append(transforms.Resize(self.min_edge_size))
            if img_transforms.RANDOM_CROP.enable:
                transform_list.append(transforms.RandomCrop(self.min_edge_size))
            elif img_transforms.CENTER_CROP.enable:
                transform_list.append(transforms.CenterCrop(self.min_edge_size))

        # ColorJitter
        if img_transforms.COLOR_JITTER.enable:
            params = {key: img_transforms.COLOR_JITTER[key] for key in img_transforms.COLOR_JITTER 
                            if key != 'enable'}
            transform_list.append(transforms.ColorJitter(**params))
            # brightness = img_transforms.COLOR_JITTER['brightness']
            # contrast = img_transforms.COLOR_JITTER['contrast']
            # saturation = img_transforms.COLOR_JITTER['saturation']
            # hue = img_transforms.COLOR_JITTER['hue']
            # transform_list.append(transforms.ColorJitter(brightness, contrast, saturation, hue))

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
        return transforms.Compose(transform_list)

    def check_conflict_options(self):
        count = self.cfg.TRANSFORMS.IMG.RANDOM_RESIZED_CROP.enable + \
                self.cfg.TRANSFORMS.IMG.RESIZE.enable
        assert count <= 1, 'You can only use one resize transform operation'

        count = self.cfg.TRANSFORMS.IMG.RANDOM_CROP.enable + \
                self.cfg.TRANSFORMS.IMG.CENTER_CROP.enable
        assert count <= 1, 'You can only use one crop transform operation'


@TRANSFORMS_REGISTRY.register()
class SkinTransforms(DefaultTransforms):
    def __init__(self, cfg):
        is_train = cfg.DATASET.IS_TRAIN
        mean = cfg.TRANSFORMS.TENSOR.NORMALIZATION.mean
        std = cfg.TRANSFORMS.TENSOR.NORMALIZATION.std
        normalize = transforms.Normalize(mean, std)
        pass
        # if not is_train:
        #     transform_list = [transforms.Resize(cfg.img_size+20)]
        #     if cfg.crops == 1:
        #         transform_list.append(transforms.CenterCrop(cfg.img_size))
        #         transform_list.append(transforms.ToTensor())
        #         transform_list.append(normalize)
        #     elif cfg.crops == 5:
        #         transform_list.append(transforms.FiveCrop(cfg.img_size))
        #         transform_list.append(transforms.Lambda(lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops])))
        #     elif cfg.crops == 10:
        #         transform_list.append(transforms.TenCrop(cfg.img_size))
        #         transform_list.append(transforms.Lambda(lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops])))
            
        #     transform = transforms.Compose(transform_list)
            # return transform
            # transform = transforms.Compose([
            #     transforms.Resize((cfg.img_size, cfg.img_size)),
            #     # transforms.CenterCrop(cfg.img_size),
            #     transforms.ToTensor(),
            #     normalize
            # ])


        # if cfg.TRANSFORMS.AUG_IMAGENET:
        #     transform = transforms.Compose([
        #         transforms.Resize(cfg.img_size+1),
        #         transforms.RandomCrop(cfg.img_size),
        #         autoaugment.ImageNetPolicy(),
        #         transforms.ToTensor(),
        #         normalize
        #     ])
        # elif cfg.TRANSFORMS.AUG_CIFAR:
        #     transform = transforms.Compose([
        #         transforms.Resize(cfg.img_size+1),
        #         transforms.RandomCrop(cfg.img_size),
        #         autoaugment.CIFAR10Policy(),
        #         transforms.ToTensor(),
        #         normalize
        #     ])
        # else:
        #     transform_list = [
        #         transforms.Resize(cfg.img_size+20),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomRotation(10),
        #     ]
        #     if hasattr(cfg, 'color') and cfg.color:
        #         transform_list.append(transforms.ColorJitter(0.1, 0.1, 0.1, 0.05))
        #     if hasattr(cfg, 'crops') and cfg.crops == 1:
        #         transform_list.append(transforms.RandomCrop(cfg.img_size))
        #         transform_list.append(transforms.ToTensor())
        #         transform_list.append(normalize)
        #     elif hasattr(cfg, 'crops') and cfg.crops == 5:
        #         transform_list.append(transforms.FiveCrop(cfg.img_size))
        #         transform_list.append(transforms.Lambda(lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops])))
        #     elif hasattr(cfg, 'crops') and cfg.crops == 10:
        #         transform_list.append(transforms.TenCrop(cfg.img_size))
        #         transform_list.append(transforms.Lambda(lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops])))
            
        
        #     if hasattr(cfg, 'erase') and cfg.erase:
        #         transform_list.append(RandomErasing())
        #     if hasattr(cfg, 'cutout') and cfg.cutout:
        #         transform_list.append(Cutout(n_holes=cfg.n_holes, length=cfg.length))
        #     transform = transforms.Compose(transform_list)
        # return transform
