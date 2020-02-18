import pretrainedmodels
import torch.nn as nn

from torchline.models import META_ARCH_REGISTRY

__all__ = [
    'Pretrainedmodels',
    'Nasnetamobile',
    'Pnasnet5large',
]

model_params = {
        'nasnetamobile': [1056, 224], # in_features, img_size
        'pnasnet5large': [4320, 331],
}

class Pretrainedmodels(nn.Module):
    def __init__(self, cfg):
        super(Pretrainedmodels, self).__init__()
        model_name = cfg.model.name
        img_size = cfg.input.size[0]
        assert img_size==model_params[model_name.lower()][1], "the img_size should be {model_params[model_name.lower()][1]}"
        num_classes = cfg.model.classes
        pretrained = 'imagenet' if cfg.model.pretrained else None
        model = pretrainedmodels.__dict__[model_name.lower()](num_classes=1000, pretrained=pretrained)
        model.last_linear = nn.Linear(in_features=model_params[model_name.lower()][0], 
                                    out_features=num_classes, bias=True)
        if cfg.model.finetune:
            print(f"finetuning {type(self).__name__} model")
            for name, p in model.named_parameters():
                if 'last_linear' not in name:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
        self.model = model

    def forward(self, x):
        return self.model(x)
        
@META_ARCH_REGISTRY.register()
class Nasnetamobile(Pretrainedmodels):
    def __init__(self, cfg):
        super(Nasnetamobile, self).__init__(cfg)


@META_ARCH_REGISTRY.register()
class Pnasnet5large(Pretrainedmodels):
    def __init__(self, cfg):
        super(Pnasnet5large, self).__init__(cfg)
