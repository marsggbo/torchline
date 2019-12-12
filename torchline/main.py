"""
Runs a model on a single node across N-gpus.
"""
import sys
sys.path.append('.')
import argparse
import os
from argparse import ArgumentParser

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.logging import TestTubeLogger, MLFlowLogger

from torchline.config import get_cfg
from torchline.engine.lightning_module_template import LightningTemplateModel
from torchline.utils import Logger

logger = Logger(__name__).getlog()
SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    if hasattr(args, "config_file"):
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file, open(args.config_file, "r").read()
            )
        )
        
    # if not (hasattr(args, "eval_only") and args.eval_only):
    #     torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK
    
    logger.info("Running with full config:\n{}".format(cfg))
    return cfg

class MyTrainer(Trainer):
    def __init__(self, cfg, hparams):
        self.cfg = cfg
        self.hparams = hparams
        
        # hooks
        HOOKS = self.cfg.HOOKS
        params = {key: HOOKS.EARLY_STOPPING[key] for key in HOOKS.EARLY_STOPPING if key != 'enable'}
        early_stop_callback = EarlyStopping(**params) if HOOKS.EARLY_STOPPING['enable'] else False

        params = {key: HOOKS.MODEL_CHECKPOINT[key] for key in HOOKS.MODEL_CHECKPOINT if key != 'enable'}
        checkpoint_callback = ModelCheckpoint(**params)

        # logger
        LOGGER = self.cfg.TRAINER.LOGGER
        params = {key: LOGGER[key] for key in LOGGER if key != 'type'}
        if LOGGER['type'] == 'mlflow':
            logger = MLFlowLogger(**params)
        elif LOGGER['type'] == 'test_tube':
            logger = TestTubeLogger(**params)
        else:
            print(f"{LOGGER['type']} not supported")
            raise NotImplementedError

        # you can update trainer_params to change different parameters
        self.trainer_params = {
            'gpus': hparams.gpus,
            'use_amp': hparams.use_16bit,
            'distributed_backend': hparams.distributed_backend,

            # 'min_epochs' : cfg.TRAINER.MIN_EPOCHS,
            # 'max_epochs' : cfg.TRAINER.MAX_EPOCHS,
            'gradient_clip_val' : cfg.TRAINER.GRAD_CLIP_VAL,
            'show_progress_bar' : cfg.TRAINER.SHOW_PROGRESS_BAR,
            'row_log_interval' : cfg.TRAINER.ROW_LOG_INTERVAL,
            'log_save_interval' : cfg.TRAINER.LOG_SAVE_INTERVAL,
            'log_gpu_memory' : cfg.TRAINER.LOG_GPU_MEMORY,
            'default_save_path' : cfg.TRAINER.DEFAULT_SAVE_PATH,
            'fast_dev_run' : cfg.TRAINER.FAST_DEV_RUN,

            'logger': logger,
            'early_stop_callback': early_stop_callback,
            'checkpoint_callback': checkpoint_callback,

            'weights_summary': None
        }

        super(MyTrainer, self).__init__(
            **self.trainer_params
        )
    

def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    """

    cfg = setup(hparams)

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = LightningTemplateModel(cfg)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = MyTrainer(cfg, hparams)

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)

    # gpu args
    parent_parser.add_argument(
        "--config_file", 
        default="", 
        metavar="FILE", 
        help="path to config file"
    )
    parent_parser.add_argument(
        '--gpus',
        type=int,
        default=2,
        help='how many gpus'
    )
    parent_parser.add_argument(
        '--distributed_backend',
        type=str,
        default='dp',
        help='supports three options dp, ddp, ddp2'
    )
    parent_parser.add_argument(
        '--use_16bit',
        dest='use_16bit',
        action='store_true',
        help='if true uses 16 bit precision'
    )
    parent_parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    # each LightningModule defines arguments relevant to it
    hyperparams = parent_parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)
