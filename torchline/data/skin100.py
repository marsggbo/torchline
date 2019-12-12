#coding=utf-8

import torch
import torchvision
from torchvision.datasets import ImageFolder

from .build import DATASET_REGISTRY
from .transforms import build_transforms

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
    def __init__(self, args, is_train):
        super(Skin100Dataset, self).__init__(args.data_root)
        self.args = args
        if is_train:
            self.data_list = args.trainlist
        else:
            self.data_list = args.testlist
        self.transform = get_transform_ops(args, is_train)
        self.target_transform = None
        self.samples = load_clean_data(self.data_list, self.class_to_idx)
        self.imgs = self.samples

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)
