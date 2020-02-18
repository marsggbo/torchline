"""
Runs a model on a single node across N-gpus.
"""
import argparse
import glob
import os
import shutil
from argparse import ArgumentParser

import numpy as np
import torch
from torchline.config import get_cfg
from torchline.engine import build_module
from torchline.trainer import build_trainer
from torchline.utils import get_imgs_to_predict

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    """

    cfg = get_cfg()
    cfg.setup_cfg_with_hparams(hparams)
    # only predict on some samples
    if hasattr(hparams, "predict_only") and hparams.predict_only:
        predict_only = cfg.predict_only
        if predict_only.type == 'ckpt':
            load_params = {key: predict_only.load_ckpt[key] for key in predict_only.load_ckpt}
            model = build_module(cfg)
            ckpt_path = load_params['checkpoint_path']
            model.load_state_dict(torch.load(ckpt_path)['state_dict'])
        elif predict_only.type == 'metrics':
            load_params = {key: predict_only.load_metric[key] for key in predict_only.load_metric}
            model = build_module(cfg).load_from_metrics(**load_params)
        else:
            print(f'{cfg.predict_only.type} not supported')
            raise NotImplementedError

        model.eval()
        model.freeze() 
        images = get_imgs_to_predict(cfg.predict_only.to_pred_file_path, cfg)
        if torch.cuda.is_available():
            images['img_data'] = images['img_data'].cuda()
            model = model.cuda()
        predictions = model(images['img_data'])
        class_indices = torch.argmax(predictions, dim=1)
        for i, file in enumerate(images['img_file']):
            index = class_indices[i]
            print(f"{file} is {classes[index]}")
        return predictions.cpu()
    elif hasattr(hparams, "test_only") and hparams.test_only:
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
    parent_parser.add_argument("--config_file", default="", metavar="FILE", help="path to config file")
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
