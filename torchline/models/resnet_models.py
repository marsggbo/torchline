import torch
import torch.nn as nn
from torchvision import models

from .build import META_ARCH_REGISTRY

__all__ = [
    'Resnet18',
    'Resnet34',
    'Resnet50',
    'Resnet101',
    'Resnet152',
]

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

@META_ARCH_REGISTRY.register()
class Resnet50(nn.Module):
    def __init__(self, cfg):
        super(Resnet50, self).__init__()
        self.cfg = cfg
        self.num_classes = cfg.MODEL.CLASSES
        name = cfg.MODEL.META_ARCH
        pretrained = cfg.MODEL.PRETRAINED
        model = eval(f"models.{name.lower()}(pretrained={pretrained})")
        model.fc = Identity()
        model = list(model.children())
        self.stem = nn.Sequential(*model[:4])
        self.layer1 = nn.Sequential(*model[4])
        self.layer2 = nn.Sequential(*model[5])
        self.layer3 = nn.Sequential(*model[6])
        self.layer4 = nn.Sequential(*model[7])

        self.clf = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, self.num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
    
    def extract_features(self, x):
        assert len(self.cfg.MODEL.FEATURES) >= 1
        features = {}
        features['stem'] = self.stem(x)
        features['f1'] = self.layer1(features['stem'])
        features['f2'] = self.layer2(features['f1'])
        features['f3'] = self.layer3(features['f2'])
        features['f4'] = self.layer4(features['f3'])
        if len(self.cfg.MODEL.FEATURES) == 1:
            final_feature = features[self.cfg.MODEL.FEATURES[0]]
        else:
            final_feature = features[self.cfg.MODEL.FEATURES[0]]
            for f in self.cfg.MODEL.FEATURES[1:]:
                final_feature += features[f]
        return final_feature

    
    def forward(self, x):
        '''
        params: 
            x (tensor): N*c*h*w

        return:
            img_cls_preds (tensor): N*classes
        '''
        bs= x.shape[0]
        features = self.extract_features(x)
        img_cls_preds = self.clf(features).view(bs, -1)

        return img_cls_preds



@META_ARCH_REGISTRY.register()
class Resnet18(Resnet50):
    def __init__(self, cfg):
        super(Resnet18, self).__init__(cfg)


@META_ARCH_REGISTRY.register()
class Resnet34(Resnet50):
    def __init__(self, cfg):
        super(Resnet34, self).__init__(cfg)


@META_ARCH_REGISTRY.register()
class Resnet101(Resnet50):
    def __init__(self, cfg):
        super(Resnet101, self).__init__(cfg)


@META_ARCH_REGISTRY.register()
class Resnet152(Resnet50):
    def __init__(self, cfg):
        super(Resnet152, self).__init__(cfg)
