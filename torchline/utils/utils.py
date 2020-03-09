import glob
import os

import torch
from PIL import Image
from torchvision import transforms

__all__ = [
    'image_loader',
    'get_imgs_to_predict',
    'topk_acc',
    'model_size'
]

def image_loader(filename, cfg):
    '''load an image and convert it to tensor
    Args:
        filename: image filename
        cfg: CfgNode
    return:
        torch.tensor
    '''
    image = Image.open(filename).convert('RGB')
    mean = cfg.transforms.tensor.normalization.mean
    std = cfg.transforms.tensor.normalization.std
    img_size = cfg.input.size

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
    if os.path.isfile(path):
        images = image_loader(path, cfg).unsqueeze(0)
    elif os.path.isdir(path):
        image_types = [os.path.join(path,'*.jpg'), os.path.join(path,'*.png')]
        image_files = []
        images = {
            'img_file': [],
            'img_data': []
        }
        for img_type in image_types:
            image_files.extend(glob.glob(img_type))
        for img_file in image_files:
            images['img_file'].append(img_file)
            images['img_data'].append(image_loader(img_file, cfg))
        images['img_data'] = torch.stack(images['img_data'])
    return images

def model_size(model):
    return sum([p.numel() for p in model.parameters()])*4/1024**2

def topk_acc(output, target, topk=(1, 3)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k*100.0/len(target.view(-1)))
        # res.append(correct_k.mul_(100.0 / batch_size))
    return torch.tensor(res)
