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
        self.num_classes = cfg.model.classes
        name = cfg.model.META_ARCH
        pretrained = cfg.model.pretrained
        model = eval(f"models.{name.lower()}(pretrained={pretrained})")
        model.fc = Identity()
        model = list(model.children())
        self.stem = nn.Sequential(*model[:4])
        self.layer1 = nn.Sequential(*model[4])
        self.layer2 = nn.Sequential(*model[5])
        self.layer3 = nn.Sequential(*model[6])
        self.layer4 = nn.Sequential(*model[7])

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.clf = nn.Linear(in_features=2048, out_features=self.num_classes)
    
    def extract_features(self, x):
        assert len(self.cfg.model.features) >= 1
        features = {}
        features['stem'] = self.stem(x)
        features['f1'] = self.layer1(features['stem'])
        features['f2'] = self.layer2(features['f1'])
        features['f3'] = self.layer3(features['f2'])
        features['f4'] = self.layer4(features['f3'])
        if len(self.cfg.model.features) == 1:
            final_feature = features[self.cfg.model.features[0]]
        else:
            final_feature = features[self.cfg.model.features[0]]
            for f in self.cfg.model.features[1:]:
                final_feature += features[f]
        return final_feature

    
    def forward(self, x):
        '''
        params: 
            x (tensor): N*c*h*w

        return:
            predictions (tensor): N*classes
        '''
        bs= x.shape[0]
        features = self.extract_features(x)
        predictions = self.avg_pool(features).view(bs, -1)
        predictions = self.clf(predictions).view(bs, -1)

        return predictions



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
