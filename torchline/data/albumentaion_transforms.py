from albumentations import (CLAHE, Blur, ChannelDropout, ChannelShuffle,
                            Compose, Cutout, Flip, GaussNoise, GridDistortion,
                            HueSaturationValue, IAAAdditiveGaussianNoise,
                            IAAEmboss, MotionBlur, Normalize, OneOf,
                            OpticalDistortion, RandomBrightnessContrast,
                            RandomGridShuffle, Resize, ShiftScaleRotate)
from albumentations.pytorch.transforms import ToTensor

from .transforms import TRANSFORMS_REGISTRY, BaseTransforms

__all__ = [
    'AlbumentationsTransforms'
]

@TRANSFORMS_REGISTRY.register()
class AlbumentationsTransforms(BaseTransforms):
    def __init__(self, cfg):
        super(AlbumentationsTransforms, self).__init__(cfg)
        self.is_train = cfg.dataset.is_train
        self.img_size = cfg.input.size
        self.mean = cfg.transforms.tensor.normalization.mean
        self.std = cfg.transforms.tensor.normalization.std
        self.normalize = Normalize(self.mean, self.std)
        self.init_transforms = self.get_transform()
        self.transform = self.parse_transform
    
    def parse_transform(self, image):
        image = self.init_transforms(image=image)['image']
        return image

    @property
    def valid_transform(self):
        height, width = self.img_size
        return Compose([Resize(height, width),
            self.normalize,
            ToTensor()])

    @property
    def train_transform(self):
        # albumentations transforms cfg
        at_cfg = self.cfg.abtfs
        height, width = self.img_size
        transforms_list = [Resize(height, width), Flip()]
        
        # random_grid_shuffle
        if at_cfg.random_grid_shuffle.enable:
            grid = at_cfg.random_grid_shuffle.grid
            grid = (grid,grid)
            transforms_list.append(RandomGridShuffle((grid)))

        # channel_shuffle
        if at_cfg.channel_shuffle.enable:
            transforms_list.append(ChannelShuffle(p=1))
        
        # channel_dropout
        if at_cfg.channel_dropout.enable:
            drop_range = at_cfg.channel_dropout.drop_range
            fill_value = at_cfg.channel_dropout.fill_value
            transforms_list.append(ChannelDropout(drop_range, fill_value, p=1))

        # noise
        if at_cfg.noise.enable:
            transforms_list.append(OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
            ], p=1))

        # blur
        if at_cfg.blur.enable:
            transforms_list.append(OneOf([
                MotionBlur(),
                Blur(blur_limit=3,),
            ], p=1))

        # rotate
        if at_cfg.rotate.enable:
            transforms_list.append(ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2,
                                                    rotate_limit=45, p=1))

        # distortion
        if at_cfg.distortion.enable:
            transforms_list.append(OneOf([
                OpticalDistortion(p=0.3),
                GridDistortion(p=.3),
            ], p=1))

        # bright
        if at_cfg.bright.enable:
            transforms_list.append(
                OneOf([
                    CLAHE(clip_limit=2),
                    RandomBrightnessContrast(p=0.8),            
                ], p=1))

        # hue color
        if at_cfg.hue.enable:
            transforms_list.append(HueSaturationValue(p=0.3))
        
        # cutout
        if at_cfg.cutout.enable:
            num_holes = at_cfg.cutout.num_holes
            size = at_cfg.cutout.size
            fill_value = at_cfg.cutout.fill_value
            transforms_list.append(Cutout(num_holes, size, size, fill_value, 1))
        transforms_list.append(self.normalize)
        transforms_list.append(ToTensor())

        return Compose(transforms_list)
