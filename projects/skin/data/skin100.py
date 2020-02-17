#coding=utf-8

import torch
import torchvision
from torchvision.datasets import ImageFolder

from torchline.data import build_transforms, DATASET_REGISTRY

__all__ = [
    'Skin100Dataset'
]

def load_clean_data(path, class_to_idx):
    imgs = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            class_name = line.rstrip().split('/')[-2]
            id_index = class_to_idx[class_name]
            imgs.append((line.rstrip(), id_index))
    return imgs

@DATASET_REGISTRY.register()
class Skin100Dataset(ImageFolder):
    def __init__(self, cfg):
        super(Skin100Dataset, self).__init__(cfg.dataset.dir)
        self.cfg = cfg
        is_train = self.cfg.dataset.is_train
        if is_train:
            self.data_list = cfg.dataset.train_list
        else:
            self.data_list = cfg.dataset.test_list
        self.transforms = build_transforms(cfg)
        self.target_transform = None
        self.samples = load_clean_data(self.data_list, self.class_to_idx)
        self.imgs = self.samples

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transforms is not None:
            sample = self.transforms.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)
