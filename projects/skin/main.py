"""
Runs a model on a single node across N-gpus.
"""
import sys
sys.path.append('.')
import argparse
import os
import glob
from argparse import ArgumentParser

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.logging import TestTubeLogger, MLFlowLogger

from torchline.config import get_cfg
from torchline.engine import build_module_template
from torchline.utils import Logger, get_imgs_to_predict

from models import *
from data import *
from config import add_skin_config
from losses import FocalLoss

logger_print = Logger(__name__).getlog()

def parse_cfg_param(cfg_item):
    return cfg_item if cfg_item else None

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    add_skin_config(cfg)
    cfg.merge_from_list(args.opts)
    cfg.update({'hparams': args})
    cfg.freeze()
    
    if hasattr(args, "config_file"):
        logger_print.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file, open(args.config_file, "r").read()
            )
        )
    
    SEED = cfg.SEED
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if not (hasattr(args, "test_only") and args.test_only):
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = cfg.DEFAULT_CUDNN_BENCHMARK

    
    logger_print.info("Running with full config:\n{}".format(cfg))
    return cfg

class MyTrainer(Trainer):
    def __init__(self, cfg, hparams):
        self.cfg = cfg
        self.hparams = hparams
        resume_from_checkpoint = None 
        if hparams.resume:
            resume_from_checkpoint = hparams.resume
        
        # logger
        LOGGER = self.cfg.TRAINER.LOGGER
        if LOGGER.type == 'mlflow':
            params = {key: LOGGER.MLFLOW[key] for key in LOGGER.MLFLOW}
            custom_logger = MLFlowLogger(**params)
        elif LOGGER.type == 'test_tube':
            params = {key: LOGGER.TEST_TUBE[key] for key in LOGGER.TEST_TUBE}
            save_dir = cfg.TRAINER.DEFAULT_SAVE_PATH
            path = os.path.join(save_dir, LOGGER.TEST_TUBE.name)
            if not os.path.exists(path):
                version = 0
            else:
                paths = os.listdir(path)
                version = len(paths)
            if LOGGER.SETTING==2: params.update({'version': version, 'save_dir': save_dir})
            custom_logger = TestTubeLogger(**params)
        else:
            print(f"{LOGGER.type} not supported")
            raise NotImplementedError
        
        loggers = {
            0: True,
            1: False,
            2: custom_logger
        } # 0: True (default)  1: False  2: custom
        logger = loggers[LOGGER.SETTING] 

        # hooks
        HOOKS = self.cfg.HOOKS
        params = {key: HOOKS.EARLY_STOPPING[key] for key in HOOKS.EARLY_STOPPING if key != 'type'}
        early_stop_callbacks = {
            0: True,  # default setting
            1: False, # do not use early stopping
            2: EarlyStopping(**params) # use custom setting
        }
        assert HOOKS.EARLY_STOPPING.type in early_stop_callbacks, 'The type of early stopping can only be in [0,1,2]'
        early_stop_callback = early_stop_callbacks[HOOKS.EARLY_STOPPING.type]

        params = {key: HOOKS.MODEL_CHECKPOINT[key] for key in HOOKS.MODEL_CHECKPOINT if key != 'type'}
        if HOOKS.MODEL_CHECKPOINT.type==2 and LOGGER.SETTING==2:
            filepath = os.path.join(cfg.TRAINER.DEFAULT_SAVE_PATH, logger.name,
                            f'version_{logger.version}','checkpoints')
            params.update({'filepath': filepath})
        checkpoint_callbacks = {
            0: True,
            1: False,
            2: ModelCheckpoint(**params)
        }
        assert HOOKS.MODEL_CHECKPOINT.type in checkpoint_callbacks, 'The type of model checkpoint_callback can only be in [0,1,2]'
        checkpoint_callback = checkpoint_callbacks[HOOKS.MODEL_CHECKPOINT.type]
        

        # you can update trainer_params to change different parameters
        self.trainer_params = {
            'gpus': hparams.gpus,
            'use_amp': hparams.use_16bit,
            'distributed_backend': hparams.distributed_backend,

            'min_epochs' : cfg.TRAINER.MIN_EPOCHS,
            'max_epochs' : cfg.TRAINER.MAX_EPOCHS,
            'gradient_clip_val' : cfg.TRAINER.GRAD_CLIP_VAL,
            'show_progress_bar' : cfg.TRAINER.SHOW_PROGRESS_BAR,
            'row_log_interval' : cfg.TRAINER.ROW_LOG_INTERVAL,
            'log_save_interval' : cfg.TRAINER.LOG_SAVE_INTERVAL,
            'log_gpu_memory' : parse_cfg_param(cfg.TRAINER.LOG_GPU_MEMORY),
            'default_save_path' : cfg.TRAINER.DEFAULT_SAVE_PATH,
            'fast_dev_run' : cfg.TRAINER.FAST_DEV_RUN,

            'logger': logger,
            'early_stop_callback': early_stop_callback,
            'checkpoint_callback': checkpoint_callback,

            'weights_summary': None,
            'resume_from_checkpoint': resume_from_checkpoint
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

    if hasattr(hparams, "test_only") and hparams.test_only:
        load_params = {key: cfg.TEST_ONLY[key] \
                    for key in cfg.TEST_ONLY \
                    if key not in ['test_file_paths', 'type']}
        if cfg.TEST_ONLY.type == 'ckpt':
            model = build_module_template(cfg).load_from_checkpoint(**load_params)
        elif cfg.TEST_ONLY.type == 'metrics':
            model = build_module_template(cfg).load_from_metrics(**load_params)
        else:
            print(f'{cfg.TEST_ONLY.type} not supported')
            raise NotImplementedError

        model.eval()
        model.freeze() 
        images = get_imgs_to_predict(cfg.TEST_ONLY.test_file_paths, cfg)
        if torch.cuda.is_available():
            images = images.cuda()
            model = model.cuda()
        predictions = model(images)
        return predictions.cpu()
    else:
        model = build_module_template(cfg)
        trainer = MyTrainer(cfg, hparams)
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
        '--resume',
        type=str,
        default='',
        help='resume_from_checkpoint'
    )
    parent_parser.add_argument(
        '--use_16bit',
        dest='use_16bit',
        action='store_true',
        help='if true uses 16 bit precision'
    )
    parent_parser.add_argument(
        '--test_only',
        action='store_true',
        help='if true run trainer.test(model)'
    )
    parent_parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    # each LightningModule defines arguments relevant to it
    hparams = parent_parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hparams)
