
import torch
from torchvision import transforms
from torchline.data import TRANSFORMS_REGISTRY, DefaultTransforms

@TRANSFORMS_REGISTRY.register()
class SkinTransforms(DefaultTransforms):
    def __init__(self, cfg):
        super(SkinTransforms, self).__init__(cfg)
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
