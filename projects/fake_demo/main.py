"""
Runs a model on a single node across N-gpus.
"""
import argparse
import glob
import os
import shutil
import sys
sys.path.append('../..')
from argparse import ArgumentParser

import numpy as np
import torch

from torchline.models import META_ARCH_REGISTRY
from torchline.trainer import build_trainer
from torchline.config import get_cfg
from torchline.engine import build_module
from torchline.utils import Logger, get_imgs_to_predict

@META_ARCH_REGISTRY.register()
class FakeNet(torch.nn.Module):
    def __init__(self, cfg):
        super(FakeNet, self).__init__()
        self.cfg = cfg
        h, w = cfg.input.size
        self.feat = torch.nn.Conv2d(3,16,3,1,1)
        self.clf = torch.nn.Linear(16, cfg.model.classes)
        print(self)

    def forward(self, x):
        b = x.shape[0]
        out = self.feat(x)
        out = torch.nn.AdaptiveAvgPool2d(1)(out).view(b, -1)
        out = self.clf(out)
        return out

def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    """

    cfg = get_cfg()
    cfg.setup_cfg_with_hparams(hparams)
    if hasattr(hparams, "test_only") and hparams.test_only:
        model = build_module(cfg)
        trainer = build_trainer(cfg, hparams)
        trainer.test(model)
    else:
        model = build_module(cfg)
        trainer = build_trainer(cfg, hparams)
        trainer.fit(model)


if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)

    # gpu args
    parent_parser.add_argument("--config_file", default="./fake_cfg.yaml", metavar="FILE", help="path to config file")
    parent_parser.add_argument('--test_only', action='store_true', help='if true, return trainer.test(model). Validates only the test set')
    parent_parser.add_argument('--predict_only', action='store_true', help='if true run model(samples). Predict on the given samples.')
    parent_parser.add_argument( "opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

    # each LightningModule defines arguments relevant to it
    hparams = parent_parser.parse_args()
    assert not (hparams.test_only and hparams.predict_only), "You can't set both 'test_only' and 'predict_only' True"

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hparams)
