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
def AlbumentationsTransforms(cfg):
    is_train = cfg.dataset.is_train
    log_name = cfg.log.name
    mean = cfg.transforms.tensor.normalization.mean
    std = cfg.transforms.tensor.normalization.std
    img_size = cfg.input.size
    random_grid_shuffle = cfg.abtfs.random_grid_shuffle
    channel_shuffle = cfg.abtfs.channel_shuffle
    channel_dropout = cfg.abtfs.channel_dropout
    noise = cfg.abtfs.noise
    blur = cfg.abtfs.blur
    distortion = cfg.abtfs.distortion
    bright = cfg.abtfs.bright
    hue = cfg.abtfs.hue
    cutout = cfg.abtfs.cutout
    rotate = cfg.abtfs.rotate
    return _AlbumentationsTransforms(is_train, log_name, img_size, 
            random_grid_shuffle,
            channel_shuffle,
            channel_dropout,
            noise,
            blur,
            distortion,
            bright,
            hue,
            cutout,
            rotate,
            mean, std)

class _AlbumentationsTransforms(BaseTransforms):
    def __init__(self, is_train, log_name, img_size,
                 random_grid_shuffle={'enable':0, 'grid':3},
                 channel_shuffle={'enable':0, 'p':1},
                 channel_dropout={'enable':0, 'drop_range':(1, 1), 'fill_value':127},
                 noise={'enable':0, 'p':1},
                 blur={'enable':0, 'p':1, 'blur_limit':3},
                 distortion={'enable':0, 'p':1},
                 bright={'enable':0, 'p':1, 'clip_limit':2},
                 hue={'enable':0, 'p':1},
                 cutout={'enable':0, 'p':1, 'size':16, 'fill_value':127},
                 rotate={'enable':0, 'p':1, 'shift_limit': 0.0625, 'scale_limit':0.2, 'rotate_limit':45},
                 mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]):
        super(_AlbumentationsTransforms, self).__init__(is_train, log_name)
        self.is_train = is_train
        self.img_size = img_size
        self.mean = mean
        self.std = std
        self.normalize = Normalize(self.mean, self.std)
        self.random_grid_shuffle = random_grid_shuffle
        self.channel_shuffle = channel_shuffle
        self.channel_dropout = channel_dropout
        self.noise = noise
        self.blur = blur
        self.distortion = distortion
        self.bright = bright
        self.hue = hue
        self.cutout = cutout
        self.rotate = rotate
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
        height, width = self.img_size
        transforms_list = [Resize(height, width), Flip()]
        
        # random_grid_shuffle
        if self.random_grid_shuffle['enable']:
            grid = self.random_grid_shuffle['grid']
            grid = (grid,grid)
            transforms_list.append(RandomGridShuffle((grid)))

        # channel_shuffle
        if self.channel_shuffle['enable']:
            transforms_list.append(ChannelShuffle(p=1))
        
        # channel_dropout
        if self.channel_dropout['enable']:
            drop_range = self.channel_dropout.drop_range
            fill_value = self.channel_dropout.fill_value
            transforms_list.append(ChannelDropout(drop_range, fill_value, p=1))

        # noise
        if self.noise['enable']:
            transforms_list.append(OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
            ], p=1))

        # blur
        if self.blur['enable']:
            transforms_list.append(OneOf([
                MotionBlur(),
                Blur(blur_limit=3,),
            ], p=1))

        # rotate
        if self.rotate['enable']:
            params = {key:value for key, value in self.rotate.items() if key != 'enable' }
            transforms_list.append(ShiftScaleRotate(**params))

        # distortion
        if self.distortion['enable']:
            transforms_list.append(OneOf([
                OpticalDistortion(p=0.3),
                GridDistortion(p=.3),
            ], p=1))

        # bright
        if self.bright['enable']:
            transforms_list.append(
                OneOf([
                    CLAHE(clip_limit=self.bright['clip_limit']),
                    RandomBrightnessContrast(p=0.8),            
                ], p=1))

        # hue color
        if self.hue['enable']:
            transforms_list.append(HueSaturationValue(p=0.3))
        
        # cutout
        if self.cutout['enable']:
            num_holes = self.cutout['num_holes']
            size = self.cutout['size']
            fill_value = self.cutout['fill_value']
            transforms_list.append(Cutout(num_holes, size, size, fill_value, 1))
        transforms_list.append(self.normalize)
        transforms_list.append(ToTensor())

        return Compose(transforms_list)
