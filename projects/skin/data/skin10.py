#coding=utf-8

import torch
import torchvision
from torchvision.datasets import ImageFolder
from torchline.data import build_transforms, DATASET_REGISTRY

__all__ = [
    'Skin10Dataset'
]

def load_clean_data(path, class_to_idx):
    imgs = []
    convert_index = {
    #'Acne_Vulgaris':
    1 : 0,
    #'Actinic_solar_Damage_Actinic_Keratosis':
    5 : 1,
    #'Atopic_Dermatitis_Eczema':
    16: 2,
    #'Basal_Cell_Carcinoma':
    17: 3,
    #'Compound_Nevus':
    22: 4,
    #'Onychomycosis':
    66: 5,
    #'Rosacea':
    75: 6,
    #'Seborrheic_Keratosis':
    81: 7,
    #'Stasis_Ulcer':
    85: 8,
    #'Tinea_Corporis':
    90: 9,
    }
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            class_name = line.rstrip().split('/')[-2]
            id_index = class_to_idx[class_name]
            id_index = convert_index[id_index]
            imgs.append((line.rstrip(), id_index))
    return imgs

@DATASET_REGISTRY.register()
class Skin10Dataset(ImageFolder):
    def __init__(self, cfg):
        super(Skin10Dataset, self).__init__(cfg.DATASET.DIR)
        self.cfg = cfg
        is_train = self.cfg.DATASET.IS_TRAIN
        if is_train:
            self.data_list = cfg.DATASET.TRAIN_LIST
        else:
            self.data_list = cfg.DATASET.TEST_LIST
        self.transforms = build_transforms(cfg)
        self.target_transforms = None
        self.samples = load_clean_data(self.data_list, self.class_to_idx)
        self.imgs = self.samples

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transforms is not None:
            sample = self.transforms.transform(sample)
        if self.target_transforms is not None:
            target = self.target_transforms(target)

        return sample, target

    def __len__(self):
        return len(self.samples)
