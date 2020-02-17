import sys
sys.path.append('../')
import torch
import torchline
from torchline.models import build_model
from torchline.config import get_cfg

import argparse

Args = argparse.ArgumentParser()
# Resnet series: 18,34,50,101,152
# PNASNet series: A, B
# EfficientNet serirs: B0
Args.add_argument('--model', default='Resnet50')
Args.add_argument('--img_size', default=32, type=int)
args = Args.parse_args()


cfg = get_cfg()
cfg.model.meta_arch = args.model
cfg.model.classes = 10
model = build_model(cfg)
print(type(model).__name__)
size = args.img_size
y = model(torch.randn(1,3,size,size))
print(y.size())