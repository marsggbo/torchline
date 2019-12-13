import os

import torch
from PIL import Image
from torchvision import transforms


def image_loader(filename, cfg):
    '''load an image and convert it to tensor
    Args:
        filename: image filename
        cfg: CfgNode
    return:
        torch.tensor
    '''
    image = Image.open(filename).convert('RGB')
    mean = cfg.TRANSFORMS.NORMALIZATION.MEAN
    std = cfg.TRANSFORMS.NORMALIZATION.STD
    img_size = cfg.INPUT.SIZE

    transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
    image = transform(image)
    return image

def get_imgs_to_predict(path, cfg):
    ''''load images which are only used for prediction or testing
    Args:
        path: str
    return:
        torch.tensor (N*C*H*W)
    '''
    if os.path.isfile(path, cfg):
        images = image_loader(path, cfg).unsqueeze(0)
    elif os.path.isdir(path, cfg):
        images = []
        for _path in path:
            images.append(image_loader(_path, cfg))
        images = torch.stack(images)
    return images